"""Cloud LLM baseline runner (OpenAI-compatible).

This module is optional: it calls a remote model to produce baseline outputs
for `entropy_demo.eval`. It does not affect the cognitive core.

Usage (PowerShell):
  $env:OPENAI_API_KEY="..."   # do this in your shell, not in code
  python -m entropy_demo.baseline_pack --tasks data/tasks.json --mode isolated
  python -m entropy_demo.baseline_llm --in .run/baseline_prompt_pack.jsonl --out .run/baseline_outputs.jsonl --model <model> --base-url <openai-compatible-url>
  python -m entropy_demo.eval --tasks data/tasks.json --mode isolated --baseline-jsonl .run/baseline_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _coerce_baseline_json(
    raw_text: str,
    *,
    task_id: str,
    allowed_ids: set[str],
) -> dict[str, Any]:
    """Parse model output into the evaluator JSONL schema, with guardrails."""

    text = raw_text.strip()
    obj: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            obj = parsed
    except Exception:
        m = _JSON_OBJ_RE.search(text)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                obj = None

    if obj is None:
        return {
            "task_id": task_id,
            "refuse": True,
            "answer": "",
            "reason": "parse_failed",
            "used_ids": [],
            "raw_output": raw_text[:2000],
        }

    refuse = obj.get("refuse")
    if isinstance(refuse, str):
        refuse = refuse.strip().lower() in ("true", "yes", "refuse", "reject")
    refuse = bool(refuse) if isinstance(refuse, (bool, int)) else False

    used_ids = obj.get("used_ids", [])
    if isinstance(used_ids, str):
        used_ids = [used_ids]
    if not isinstance(used_ids, list):
        used_ids = []
    used_ids = [str(x) for x in used_ids if isinstance(x, (str, int, float))]

    # Enforce that citations only reference S1-provided ids.
    used_ids = [x for x in used_ids if x in allowed_ids]

    answer = obj.get("answer", "")
    reason = obj.get("reason", "")
    if not isinstance(answer, str):
        answer = ""
    if not isinstance(reason, str):
        reason = ""

    # If model claims to answer but provides no valid citations, flip to refuse.
    if not refuse and not used_ids:
        refuse = True
        reason = (reason + "; " if reason else "") + "no_valid_used_ids"
        answer = ""

    return {
        "task_id": task_id,
        "refuse": bool(refuse),
        "answer": answer,
        "reason": reason,
        "used_ids": used_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible cloud model as a baseline on the prompt pack")
    parser.add_argument("--in", dest="inp", type=str, default=".run/baseline_prompt_pack.jsonl", help="Input prompt-pack JSONL path")
    parser.add_argument("--out", type=str, default=".run/baseline_llm_outputs.jsonl", help="Output baseline JSONL path")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., deepseek-v3-2-251201)")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible base URL (required for most non-OpenAI providers)")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Env var name for API key")
    parser.add_argument(
        "--api-key-file",
        type=str,
        default="",
        help="Optional path to a file containing the API key (recommended: a gitignored file under .run/)",
    )
    parser.add_argument("--system", type=str, default="你是一个严格的 JSON 生成器。你必须只输出一个 JSON 对象，不得输出任何多余字符。", help="System prompt")
    parser.add_argument("--max", type=int, default=0, help="Max tasks to run (0 means all)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    args = parser.parse_args()

    api_key = ""
    if args.api_key_file:
        api_key = Path(str(args.api_key_file)).read_text(encoding="utf-8").strip()
    if not api_key:
        api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key (set env var {args.api_key_env} or pass --api-key-file)")

    pack = _read_jsonl(Path(args.inp))
    if not pack:
        raise SystemExit(f"no records in {args.inp}")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    out: list[dict[str, Any]] = []
    n = 0
    for rec in pack:
        if args.max and n >= int(args.max):
            break
        task_id = str(rec.get("task_id", ""))
        prompt = rec.get("prompt", "")
        if not task_id or not isinstance(prompt, str) or not prompt.strip():
            continue

        s1 = rec.get("s1", {})
        allowed_ids = set()
        if isinstance(s1, dict):
            ids = s1.get("retrieved_ids", [])
            if isinstance(ids, list):
                allowed_ids = {str(x) for x in ids}

        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": str(args.system)},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        out.append(_coerce_baseline_json(text, task_id=task_id, allowed_ids=allowed_ids))

        n += 1
        if args.sleep:
            time.sleep(float(args.sleep))

    _write_jsonl(Path(args.out), out)
    print(json.dumps({"ran": int(n), "out": args.out}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
