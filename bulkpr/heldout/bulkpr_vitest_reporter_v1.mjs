// bulkpr structured gate evidence reporter for vitest.
// Four required elements:
//   - TestCase.id as primary key
//   - runtime and typecheck entries kept separate (meta().typecheck discriminates;
//     confirmed on vitest 4.1.5)
//   - three error lists kept separate: typecheck_case_errors,
//     typecheck_source_errors (type === "Unhandled Source Error", pinned to 4.1.5 source),
//     and unhandled_errors
//   - self-attestation (schema + version + own absolute path)
// Observer-only: does not interfere with the test run.
// Atomic write: tmp + fsync + rename.
// hooks and underlying module errors are collected via vitest.state.idMap
// (confirmed usable via live object probing; set to null/empty when unavailable —
// classify handles "failed with no diagnostics" via an independent INFRA guard in
// gate_vitest.classify_vitest).
import { writeFileSync, fsyncSync, openSync, closeSync, renameSync } from "node:fs";
import { fileURLToPath } from "node:url";

const SCHEMA_VERSION = "bulkpr-vitest-report-v1";
const REPORTER_VERSION = "1.0.0";

function ser(e) {
  return { message: String(e?.message ?? e ?? ""), name: e?.name ?? null,
           type: e?.type ?? null,
           stacks: Array.isArray(e?.stacks)
             ? e.stacks.map(s => ({ file: s?.file ?? null, line: s?.line ?? null,
                                    column: s?.column ?? null })) : null,
           stack_head: (e?.stack ?? "").split("\n").slice(0, 3).join("\n") };
}

export default class BulkprGateReporter {
  onInit(vitest) { this.vitest = vitest; }
  onTestRunEnd(testModules, unhandledErrors, reason) {
    const cases = [], moduleErrors = [], tcCaseErrors = [], modules = [];
    const fileOrder = [], testOrderByFile = {}, resolvedConfigs = {};
    for (const mod of testModules) {
      const file = mod.moduleId;
      const project = mod.project?.name ?? null;
      const kind = mod.meta()?.typecheck ? "typecheck" : "runtime";
      const mkey = `${project}|${file}|${kind}`;
      fileOrder.push(mkey);
      modules.push({ file, project, kind, state: mod.state() });
      if (project != null && !(project in resolvedConfigs)) {
        // Keys confirmed readable from the serialized project config via live probing;
        // maxWorkers/fileParallelism are not present in the serialized form → null,
        // covered by config file bytes in the fingerprint as a fallback.
        const c = mod.project?.config ?? {};
        resolvedConfigs[project] = {
          pool: c.pool ?? null, maxConcurrency: c.maxConcurrency ?? null,
          isolate: c.isolate ?? null, sequence: c.sequence ?? null,
          fileParallelism: c.fileParallelism ?? null,
          maxWorkers: c.maxWorkers ?? null,
          typecheck_enabled: c.typecheck?.enabled ?? null };
      }
      const ids = [];
      for (const tc of mod.children.allTests()) {
        const r = tc.result();
        // hooks three-state: task found but no hooks recorded = {} (no hook involvement);
        // task not found or lookup failed = null (attribution unknown →
        // verdict core classifies a failed runtime case as INFRA)
        let hooks = null;
        try {
          const rawTask = this.vitest?.state?.idMap?.get(tc.id);
          hooks = rawTask ? (rawTask.result?.hooks ?? {}) : null;
        } catch { hooks = null; }
        const entry = { id: tc.id, project, file, kind,
                        fullName: tc.fullName, name: tc.name,
                        mode: tc.options?.mode ?? null,
                        duration: tc.diagnostic?.()?.duration ?? null,
                        state: r?.state ?? "unknown", hooks,
                        errors: (r?.errors ?? []).map(ser) };
        cases.push(entry); ids.push(tc.id);
        if (kind === "typecheck" && entry.state === "failed") tcCaseErrors.push(entry);
      }
      testOrderByFile[mkey] = ids;
      // Module-level errors: union of reported errors() and underlying task result.errors,
      // deduplicated. (Confirmed by live probing: file-level typecheck errors are visible
      // when present alone, but can be swallowed by vitest markState when they co-exist with
      // case errors — so the union may still be incomplete; the verdict core classifies
      // that situation as INFRA.)
      let rawErrs = [];
      try { rawErrs = this.vitest?.state?.idMap?.get(mod.id)?.result?.errors ?? []; }
      catch { rawErrs = []; }
      // Dedup key = full serialisation (truncated keys silently merge distinct diagnostics;
      // only the same error object appearing in both views should be merged —
      // different messages must all be preserved)
      const seen = new Set(), merged = [];
      for (const e of [...mod.errors(), ...rawErrs]) {
        const s = ser(e);
        const k = JSON.stringify([s.name, s.message, s.stacks]);
        if (!seen.has(k)) { seen.add(k); merged.push(s); }
      }
      if (merged.length) moduleErrors.push({ file, project, kind, errors: merged });
    }
    const sourceErrors = [], otherUnhandled = [];
    for (const e of unhandledErrors) {
      // "Typecheck Error" (non-zero checker exit fallback) is not a source error → goes to other
      (e?.type === "Unhandled Source Error" ? sourceErrors : otherUnhandled)
        .push(ser(e));
    }
    const report = {
      schema_version: SCHEMA_VERSION, reporter_version: REPORTER_VERSION,
      reporter_path: fileURLToPath(import.meta.url),
      run_reason: reason ?? null,
      test_cases: cases, modules, module_errors: moduleErrors,
      typecheck_case_errors: tcCaseErrors,
      typecheck_source_errors: sourceErrors, unhandled_errors: otherUnhandled,
      resolved_configs: resolvedConfigs,
      file_order: fileOrder, test_order_by_file: testOrderByFile,
    };
    const dest = process.env.BULKPR_GATE_REPORT;
    if (!dest) throw new Error("BULKPR_GATE_REPORT not set");
    const tmp = `${dest}.tmp.${process.pid}`;
    writeFileSync(tmp, JSON.stringify(report));
    const fd = openSync(tmp, "r+"); fsyncSync(fd); closeSync(fd); renameSync(tmp, dest);
  }
}
