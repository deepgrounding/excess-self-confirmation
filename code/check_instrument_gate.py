"""main.md 6.2: instrument admission gates (before H1-H3).

Checks |SEL(T)| and |ESC(T)| ~ 0 (within cluster-bootstrap noise) for a
J_oracle or J_placebo run. Reuses esc_core.decompose (the reference
SEL/ESC implementation, same object sim_validate.py checks to machine
precision) rather than reimplementing the identity.

Usage:
  python3 check_instrument_gate.py --jsonl ../data/stage2/A_pilot_oracle_math.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esc_core import decompose  # noqa: E402


def load_rounds(path: Path) -> tuple[list[dict], dict]:
    """Returns (rounds for decompose(), {task_id: {round: {J,G per cand}}})
    for bootstrap resampling by task cluster."""
    by_round_task: dict[int, dict[str, list[tuple[float, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_round_task[r["round"]][r["task_id"]].append(
                (r["j_score"], r["G"], r["selected"])
            )
    rounds_sorted = sorted(by_round_task)
    task_ids = sorted({tid for rt in by_round_task.values() for tid in rt})

    full_rounds = [pooled_for(by_round_task[t], task_ids) for t in rounds_sorted]
    return full_rounds, {"by_round_task": by_round_task, "rounds_sorted": rounds_sorted, "task_ids": task_ids}


def bootstrap_se(meta: dict, statfn, B: int = 2000, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    task_ids = meta["task_ids"]
    n = len(task_ids)
    vals = []
    for _ in range(B):
        sample = rng.choice(task_ids, size=n, replace=True).tolist()
        rounds = [
            pooled_for(meta["by_round_task"][t], sample) for t in meta["rounds_sorted"]
        ]
        d = decompose(rounds)
        vals.append(statfn(d))
    return float(np.std(vals))


def pooled_for(rt: dict, subset: list[str]) -> dict:
    J_all, G_all, J_sel, G_sel = [], [], [], []
    for tid in subset:
        for j, g, sel in rt.get(tid, []):
            J_all.append(j)
            G_all.append(g)
            if sel:
                J_sel.append(j)
                G_sel.append(g)
    return {
        "J_sel": float(np.mean(J_sel)) if J_sel else float("nan"),
        "G_sel": float(np.mean(G_sel)) if G_sel else float("nan"),
        "J_pop": float(np.mean(J_all)) if J_all else float("nan"),
        "G_pop": float(np.mean(G_all)) if G_all else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--condition", choices=["oracle", "placebo"], required=True)
    ap.add_argument("--z", type=float, default=2.0, help="pass if |stat(T)| <= z * bootstrap SE")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    rounds, meta = load_rounds(args.jsonl)
    d = decompose(rounds)
    T = len(rounds) - 1

    if args.condition == "oracle":
        checks = {"SEL(T)": (lambda dd: dd["SEL"][T], d["SEL"][T]),
                  "ESC(T)": (lambda dd: dd["ESC"][T], d["ESC"][T])}
    else:
        checks = {"ESC(T)": (lambda dd: dd["ESC"][T], d["ESC"][T])}

    print(f"{args.jsonl.name}: T={T} n_tasks={len(meta['task_ids'])}", flush=True)
    all_pass = True
    result = {"jsonl": str(args.jsonl), "condition": args.condition, "T": T,
              "n_tasks": len(meta["task_ids"])}
    for name, (statfn, val) in checks.items():
        se = bootstrap_se(meta, statfn, B=args.n_boot)
        ok = bool(abs(val) <= args.z * se)
        all_pass = all_pass and ok
        print(f"  {name}: value={val:+.4f}  bootstrap_SE={se:.4f}  "
              f"|value|<= {args.z}*SE={args.z*se:.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
        result[name] = {"value": float(val), "bootstrap_se": float(se), "pass": ok}
    result["gate_pass"] = all_pass
    print(f"GATE_RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
