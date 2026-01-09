"""Task-based evaluation for the cognitive core, with optional transformer comparison.

This evaluator is intentionally objective and simple:
- It runs the cognitive core on each task and checks deterministic expectations.
- It can compare against a baseline model (e.g., a Transformer) by loading a JSONL
  file of baseline outputs and scoring them against the same task expectations.

No heavy ML dependencies are required. Baseline outputs can be produced offline.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_CONFIG, load_model_config, model_config_from_dict, model_config_to_dict
from .engine import run_engine
from .reporting import engine_result_to_dict


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("tasks file must be a JSON object")
    return raw


def _normalize_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def _task_pass_fail(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return (passed, failures)."""

    failures: list[str] = []

    exp_success = expected.get("success")
    if isinstance(exp_success, bool):
        if bool(actual.get("success")) != exp_success:
            failures.append(f"success expected={exp_success} got={actual.get('success')}")

    exp_s2 = expected.get("s2_invoked")
    if isinstance(exp_s2, bool):
        if bool(actual.get("s2_invoked")) != exp_s2:
            failures.append(f"s2_invoked expected={exp_s2} got={actual.get('s2_invoked')}")

    must_ids = expected.get("must_include_any_ids")
    if isinstance(must_ids, list) and all(isinstance(x, str) for x in must_ids):
        present = set(actual.get("final_ids", []))
        if not any(x in present for x in must_ids):
            failures.append(f"missing any of ids={must_ids}")

    must_all_ids = expected.get("must_include_all_ids")
    if isinstance(must_all_ids, list) and all(isinstance(x, str) for x in must_all_ids):
        present = set(actual.get("final_ids", []))
        missing = [x for x in must_all_ids if x not in present]
        if missing:
            failures.append(f"missing required ids={missing}")

    must_kinds = expected.get("must_include_rewrite_kinds")
    if isinstance(must_kinds, list) and all(isinstance(x, str) for x in must_kinds):
        kinds = set(actual.get("rewrite_kinds", []))
        for k in must_kinds:
            if k not in kinds:
                failures.append(f"missing rewrite_kind={k}")

    must_pairs = expected.get("must_reconcile_pairs")
    if isinstance(must_pairs, list):
        reconciled = {_normalize_pair(a, b) for a, b in actual.get("reconciled_pairs", [])}
        for pair in must_pairs:
            if isinstance(pair, list) and len(pair) == 2:
                p = _normalize_pair(str(pair[0]), str(pair[1]))
                if p not in reconciled:
                    failures.append(f"missing reconciled_pair={list(p)}")

    must_claim_keys = expected.get("must_include_claim_keys")
    if isinstance(must_claim_keys, list) and all(isinstance(x, str) for x in must_claim_keys):
        present = set(actual.get("claim_keys_present", []))
        for k in must_claim_keys:
            if k not in present:
                failures.append(f"missing claim_key={k}")

    reason_contains = expected.get("refusal_reason_contains_any")
    if isinstance(reason_contains, list) and all(isinstance(x, str) for x in reason_contains):
        if not bool(actual.get("success")):
            reason = str(actual.get("reason", ""))
            if not any(x.lower() in reason.lower() for x in reason_contains):
                failures.append("refusal_reason does not contain any expected substrings")

    return len(failures) == 0, failures


def _extract_actual_from_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    ids_final = list(
        dict.fromkeys(
            list(trace.get("s1", {}).get("retrieved_ids", [])) + list(trace.get("s2", {}).get("added_ids", []))
        )
    )

    rewrite_kinds: list[str] = []
    steps = trace.get("s2", {}).get("steps", [])
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and isinstance(s.get("rewrite_kind"), str):
                rewrite_kinds.append(s["rewrite_kind"])

    # Reconciled pairs are extracted from constraint atoms (retrieved or added).
    reconciled_pairs: list[list[str]] = []
    for section in ("s1", "s2"):
        atoms = trace.get(section, {}).get("retrieved_atoms" if section == "s1" else "added_atoms", [])
        if not isinstance(atoms, list):
            continue
        for a in atoms:
            if not isinstance(a, dict):
                continue
            eta = a.get("eta_i", {})
            if not isinstance(eta, dict):
                continue
            pairs = eta.get("reconcile_pairs", [])
            if isinstance(pairs, list):
                for p in pairs:
                    if isinstance(p, list) and len(p) == 2:
                        reconciled_pairs.append([str(p[0]), str(p[1])])

    claim_keys_present: set[str] = set()
    for section in ("s1", "s2"):
        atoms = trace.get(section, {}).get("retrieved_atoms" if section == "s1" else "added_atoms", [])
        if not isinstance(atoms, list):
            continue
        for a in atoms:
            if not isinstance(a, dict):
                continue
            eta = a.get("eta_i", {})
            if not isinstance(eta, dict):
                continue
            ck = eta.get("claim_key", eta.get("key"))
            if isinstance(ck, str) and ck:
                claim_keys_present.add(ck)
            parsed = eta.get("parsed", {})
            if isinstance(parsed, dict) and isinstance(parsed.get("key"), str):
                claim_keys_present.add(parsed["key"])

    return {
        "success": bool(trace.get("success")),
        "reason": str(trace.get("reason", "")),
        "s2_invoked": bool(trace.get("s2", {}).get("invoked")),
        "final_ids": ids_final,
        "rewrite_kinds": rewrite_kinds,
        "reconciled_pairs": reconciled_pairs,
        "claim_keys_present": sorted(claim_keys_present),
    }


def _load_baseline_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        if not isinstance(rec, dict):
            continue
        tid = rec.get("task_id")
        if isinstance(tid, str) and tid:
            out[tid] = rec
    return out


def _baseline_decision(rec: Mapping[str, Any]) -> tuple[bool | None, str]:
    """Return (success/express, note). success=None means unknown."""

    if isinstance(rec.get("refuse"), bool):
        return (not bool(rec["refuse"])), "refuse_field"
    decision = rec.get("decision")
    if isinstance(decision, str):
        d = decision.strip().lower()
        if d in ("refuse", "reject", "no"):
            return False, "decision_field"
        if d in ("answer", "yes"):
            return True, "decision_field"

    text = rec.get("output", rec.get("text"))
    if isinstance(text, str):
        low = text.lower()
        refusal_markers = ["refuse", "cannot answer", "can't answer", "unable to answer", "i don't know", "don't know"]
        if any(m in low for m in refusal_markers):
            return False, "text_heuristic"
        return True, "text_heuristic"
    return None, "unknown"

def _atom_claim_key_from_dict(atom: Mapping[str, Any]) -> str | None:
    eta = atom.get("eta_i", {})
    if isinstance(eta, dict):
        parsed = eta.get("parsed")
        if isinstance(parsed, dict):
            k = parsed.get("key")
            if isinstance(k, str) and k:
                return k
        ck = eta.get("claim_key", eta.get("key"))
        if isinstance(ck, str) and ck:
            return ck
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tasks for entropy_demo")
    parser.add_argument("--tasks", type=str, default="data/tasks.json", help="Tasks JSON path")
    parser.add_argument("--config", type=str, default="", help="Optional base config JSON path")
    parser.add_argument("--out", type=str, default=".run/eval_report.json", help="Output JSON report path")
    parser.add_argument("--mode", choices=["isolated", "persistent"], default="isolated")
    parser.add_argument("--baseline-jsonl", type=str, default="", help="Optional baseline outputs JSONL (e.g., Transformer)")
    args = parser.parse_args()

    tasks_doc = _load_json(Path(args.tasks))
    tasks = tasks_doc.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("tasks.json must contain a non-empty 'tasks' list")

    base_cfg = load_model_config(Path(args.config)) if args.config else DEFAULT_CONFIG

    # If no explicit config is provided, apply recommended defaults from the tasks file.
    if not args.config:
        defaults = tasks_doc.get("recommended_defaults", {})
        if isinstance(defaults, dict):
            k = defaults.get("retrieval_k")
            eps = defaults.get("epsilon")
            tmax = defaults.get("tmax")
            if isinstance(k, int):
                base_cfg = replace(base_cfg, retrieval=replace(base_cfg.retrieval, k=int(k)))
            if isinstance(eps, (int, float)):
                base_cfg = replace(base_cfg, entropy=replace(base_cfg.entropy, epsilon=float(eps)))
            if isinstance(tmax, int):
                base_cfg = replace(base_cfg, rewrite=replace(base_cfg.rewrite, tmax=int(tmax)))
            ver = defaults.get("verifier")
            if isinstance(ver, dict):
                cfg_dict = model_config_to_dict(base_cfg)
                cfg_dict = _deep_merge(cfg_dict, {"verifier": ver})
                base_cfg = model_config_from_dict(cfg_dict)

    baseline: dict[str, dict[str, Any]] = _load_baseline_jsonl(Path(args.baseline_jsonl)) if args.baseline_jsonl else {}

    tmp: tempfile.TemporaryDirectory[str] | None = None
    tmpdir: Path | None = None
    if args.mode == "isolated":
        tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(tmp.name)

    per_task: list[dict[str, Any]] = []
    core_pass = 0
    baseline_pass = 0
    baseline_scored = 0
    baseline_safety_pass = 0
    baseline_safety_scored = 0

    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", ""))
        q = str(t.get("query", ""))
        expected = t.get("expected", {})
        overrides = t.get("task_overrides", {})
        if not tid or not q or not isinstance(expected, dict):
            continue

        merged = model_config_to_dict(base_cfg)
        if isinstance(overrides, dict) and overrides:
            merged = _deep_merge(merged, overrides)
        cfg = model_config_from_dict(merged)

        if args.mode == "isolated" and tmpdir is not None:
            # Avoid cross-task stability leakage by using a fresh history file per task.
            cfg = replace(
                cfg,
                stability=replace(cfg.stability, history_path=tmpdir / f"history_{tid}.json"),
                writeback=replace(cfg.writeback, enabled=False),
                capacity=replace(cfg.capacity, enabled=False),
            )

        res = run_engine(q, config=cfg, verbose=False)
        trace = engine_result_to_dict(res, projection=None, include_atoms=True)
        actual = _extract_actual_from_trace(trace)
        ok, failures = _task_pass_fail(expected, actual)
        core_pass += 1 if ok else 0

        baseline_rec = baseline.get(tid)
        baseline_ok: bool | None = None
        baseline_failures: list[str] = []
        baseline_meta: dict[str, Any] = {}
        if baseline_rec is not None:
            baseline_scored += 1
            b_success, note = _baseline_decision(baseline_rec)
            baseline_meta["decision_source"] = note
            baseline_meta["decision"] = b_success
            if isinstance(expected.get("success"), bool) and b_success is not None:
                if bool(b_success) != bool(expected["success"]):
                    baseline_failures.append(f"success expected={expected['success']} got={b_success}")
                else:
                    baseline_ok = True
            else:
                baseline_ok = None
                baseline_failures.append("cannot score baseline success (missing expected.success or baseline decision)")

            # Safety-style scoring: compare baseline decision against the
            # "should answer now" gate on the initial (q, M0) only.
            #
            # This is intentionally different from the cognitive-core success
            # condition, since the core may invoke S2 to make the state
            # expressible. For a baseline without S2, this measures whether it
            # answers when the initial state is not expressible.
            eps0 = float(cfg.entropy.epsilon)
            E0 = float(trace.get("entropy", {}).get("initial", {}).get("total", 1.0))
            v0 = bool(trace.get("verifier", {}).get("initial", {}).get("passed", True))
            should_answer_now = bool(E0 <= eps0 and v0)
            baseline_meta["should_answer_now"] = should_answer_now
            baseline_meta["E0"] = E0
            baseline_meta["epsilon"] = eps0
            baseline_meta["V0"] = v0

            if b_success is not None:
                baseline_safety_scored += 1
                if bool(b_success) == should_answer_now:
                    baseline_safety_pass += 1
                else:
                    baseline_meta["safety_mismatch"] = "unsafe_answer" if bool(b_success) else "over_refusal"

            # Evidence validation (when baseline answers).
            s1_ids = set(trace.get("s1", {}).get("retrieved_ids", []))
            must_ids = expected.get("must_include_any_ids")
            used_ids = baseline_rec.get("used_ids")
            used_ids_list: list[str] | None = None
            if isinstance(used_ids, list) and all(isinstance(x, str) for x in used_ids):
                used_ids_list = list(used_ids)

            if b_success is True and (used_ids_list is None or len(used_ids_list) == 0):
                baseline_failures.append("baseline answered but used_ids is missing/empty")

            if used_ids_list is not None:
                unknown = [x for x in used_ids_list if x not in s1_ids]
                if unknown:
                    baseline_failures.append(f"used_ids_not_in_s1_retrieval={unknown}")

            if isinstance(must_ids, list) and all(isinstance(x, str) for x in must_ids):
                if used_ids_list is not None:
                    if not any(x in set(used_ids_list) for x in must_ids):
                        baseline_failures.append(f"missing any of used_ids={must_ids}")
                else:
                    baseline_failures.append("missing baseline used_ids (cannot verify evidence)")

            must_claim_keys = expected.get("must_include_claim_keys")
            if isinstance(must_claim_keys, list) and all(isinstance(x, str) for x in must_claim_keys):
                if used_ids_list is None:
                    baseline_failures.append("missing baseline used_ids (cannot verify claim_keys)")
                else:
                    id_to_key: dict[str, str] = {}
                    atoms = trace.get("s1", {}).get("retrieved_atoms", [])
                    if isinstance(atoms, list):
                        for a in atoms:
                            if isinstance(a, dict):
                                aid = a.get("id")
                                if isinstance(aid, str) and aid:
                                    ck = _atom_claim_key_from_dict(a)
                                    if isinstance(ck, str) and ck:
                                        id_to_key[aid] = ck
                    used_keys = {id_to_key.get(x, "") for x in used_ids_list}
                    for k in must_claim_keys:
                        if k not in used_keys:
                            baseline_failures.append(f"missing claim_key via used_ids={k}")

            baseline_ok = baseline_ok is True and len(baseline_failures) == 0
            baseline_pass += 1 if baseline_ok else 0

        per_task.append(
            {
                "id": tid,
                "query": q,
                "expected": expected,
                "core": {"passed": ok, "failures": failures, "actual": actual, "trace": trace},
                "baseline": {"present": baseline_rec is not None, "passed": baseline_ok, "failures": baseline_failures, "meta": baseline_meta},
            }
        )

    summary: dict[str, Any] = {
        "tasks_total": int(len(per_task)),
        "core_pass": int(core_pass),
        "core_pass_rate": float(core_pass / len(per_task)) if per_task else 0.0,
        "baseline_scored": int(baseline_scored),
        "baseline_pass": int(baseline_pass),
        "baseline_pass_rate": float(baseline_pass / baseline_scored) if baseline_scored else None,
        "baseline_safety_scored": int(baseline_safety_scored),
        "baseline_safety_pass": int(baseline_safety_pass),
        "baseline_safety_pass_rate": float(baseline_safety_pass / baseline_safety_scored)
        if baseline_safety_scored
        else None,
        "mode": args.mode,
    }

    report = {"summary": summary, "tasks": per_task}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
