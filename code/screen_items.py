"""Round-0 item screening + difficulty/cluster bins (P2.1 Week-1).

For each candidate item, draw N=4 generations (oracle G only — no judge),
keep pass_rate in [lo, hi], take up to --keep items, tertile-bin by pass_rate,
and within each bin cluster question text into 2 groups via TF-IDF + k-means.

Example (math screen, small smoke):
  PYTHONUNBUFFERED=1 uv run --with mlx-lm --with mlx --with numpy --with datasets \\
    --with scikit-learn python screen_items.py \\
    --model mlx-community/Qwen3.5-4B-4bit --domain math --pool 40 --keep 20 \\
    --out ../data/stage2/screen_math_4b.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage2_esc import gen, load_tasks, oracle_G  # noqa: E402


def _tfidf_matrix(texts: list[str]) -> np.ndarray:
    """Row-normalized TF-IDF matrix, dependency-free and deterministic."""
    import re as _re

    docs = [_re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    vocab = [w for w, _ in sorted(df.items(), key=lambda x: (-x[1], x[0]))[:2000]]
    vidx = {w: i for i, w in enumerate(vocab)}
    n = len(docs)
    X = np.zeros((n, len(vocab)))
    for i, d in enumerate(docs):
        for w in d:
            j = vidx.get(w)
            if j is not None:
                X[i, j] += 1.0
    idf = np.log((1 + n) / (1 + np.array([df[w] for w in vocab]))) + 1.0
    X *= idf
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def balanced_split(texts: list[str]) -> list[int]:
    """Deterministic 2-way balanced split: median split along the leading
    principal direction of the TF-IDF matrix. Used when free k-means clusters
    are so unbalanced that a bin x cluster cell would fall below the power
    floor -- semantic ordering is preserved, cell sizes are equal within 1."""
    X = _tfidf_matrix(texts)
    Xc = X - X.mean(axis=0, keepdims=True)
    # leading right singular vector via a few power iterations (deterministic)
    v = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    for _ in range(50):
        v = Xc.T @ (Xc @ v)
        v /= max(float(np.linalg.norm(v)), 1e-12)
    scores = Xc @ v
    order = np.argsort(scores, kind="mergesort")
    labels = [0] * len(texts)
    for rank, i in enumerate(order):
        labels[int(i)] = 0 if rank < len(order) // 2 else 1
    return labels


def _tfidf_kmeans_np(texts: list[str], k: int, seed: int) -> list[int]:
    """Dependency-free TF-IDF + cosine k-means fallback (deterministic).
    Exists because losing a 9-hour inference run to a missing sklearn import
    is not acceptable; quality is adequate for 2-way item clustering."""
    X = _tfidf_matrix(texts)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(25):
        labels = np.argmax(X @ centers.T, axis=1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                v = X[mask].mean(axis=0)
                centers[c] = v / max(float(np.linalg.norm(v)), 1e-12)
    return [int(x) for x in labels]


def tfidf_cluster(texts: list[str], k: int = 2, seed: int = 0) -> list[int]:
    if len(texts) < k:
        return [0] * len(texts)
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        X = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(texts)
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        return [int(x) for x in labels]
    except ImportError:
        print("WARN: sklearn unavailable; numpy TF-IDF k-means fallback", flush=True)
        return _tfidf_kmeans_np(texts, k, seed)


def tertile_bins(rates: np.ndarray) -> list[str]:
    """Map pass rates to easy/mid/hard by tertiles of the *kept* set.

    With coarse probe granularity (N=4 -> rates in multiples of 0.25) the
    empirical tertiles degenerate on the few discrete values and can leave a
    bin empty; in that case bin by the discrete values directly (lowest pass
    rate -> hard), which is the natural difficulty ordering."""
    if len(rates) == 0:
        return []
    distinct = sorted({float(r) for r in rates})
    if len(distinct) <= 3:
        names = {1: ["mid"], 2: ["hard", "easy"], 3: ["hard", "mid", "easy"]}[
            len(distinct)
        ]
        mapping = {v: names[i] for i, v in enumerate(distinct)}
        return [mapping[float(r)] for r in rates]
    q1, q2 = np.quantile(rates, [1 / 3, 2 / 3])
    out = []
    for r in rates:
        if r <= q1:
            out.append("hard")
        elif r <= q2:
            out.append("mid")
        else:
            out.append("easy")
    return out


def power_check(n_items: int, n_h3_conds: int = 4, traj_per_cond: int = 12, items_per_traj: int = 8) -> dict:
    need = n_h3_conds * traj_per_cond * items_per_traj
    return {
        "n_items": n_items,
        "h3_conds": n_h3_conds,
        "traj_per_cond": traj_per_cond,
        "items_per_traj": items_per_traj,
        "slots_needed_if_no_reuse": need,
        "ok_with_item_reuse": n_items >= items_per_traj,
        "note": "H3 uses ≥12 trajectories × ≥8 items per condition; items may be reused across trajectories.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--domain", choices=["math", "code"], required=True)
    ap.add_argument("--pool", type=int, default=300, help="candidate items to probe")
    ap.add_argument("--keep", type=int, default=100, help="max kept after screen")
    ap.add_argument("--N", type=int, default=4, help="round-0 candidates per item")
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, default=None, help="optional JSONL of all probes")
    ap.add_argument(
        "--from-raw", type=Path, default=None,
        help="rebuild manifest offline from a --raw-out JSONL of a previous "
             "run (same --domain/--pool/--seed/--N); no model inference",
    )
    args = ap.parse_args()

    tasks = load_tasks(args.domain, args.pool, args.seed)

    if args.from_raw:
        raw: dict[str, dict[int, float]] = {}
        with args.from_raw.open() as f:
            for line in f:
                r = json.loads(line)
                raw.setdefault(r["task_id"], {})[int(r["cand_idx"])] = float(r["G"])
        print(f"Rebuilding offline from {args.from_raw} ({len(raw)} tasks)", flush=True)
        probed = []
        for task in tasks:
            gs_map = raw.get(task["id"])
            if not gs_map or len(gs_map) < args.N:
                print(f"WARN: incomplete raw for {task['id']}; skipped", flush=True)
                continue
            gs = [gs_map[k] for k in range(args.N)]
            probed.append({**task, "pass_rate": float(np.mean(gs)), "round0_G": gs})
        finalize(args, probed)
        return

    # Resume support: reuse any (task_id, cand_idx) probes already present in
    # --raw-out, so an interrupted or pool-expanded screen only generates the
    # missing cells. (Temperature jitter is not bit-identical across resumed
    # runs; probes are independent draws, so this does not bias the screen.)
    existing: dict[str, dict[int, float]] = {}
    if args.raw_out and args.raw_out.exists():
        with args.raw_out.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn trailing line from a crash
                existing.setdefault(r["task_id"], {})[int(r["cand_idx"])] = float(r["G"])
        if existing:
            n_prev = sum(len(v) for v in existing.values())
            print(f"Resuming: {n_prev} probes already in {args.raw_out}", flush=True)

    need_gen = any(
        k not in existing.get(t["id"], {}) for t in tasks for k in range(args.N)
    )
    model = tokenizer = None
    if need_gen:
        from mlx_lm import load  # lazy: not needed for --from-raw / full resume

        print(f"Loading {args.model}; pool={len(tasks)}", flush=True)
        model, tokenizer = load(args.model)
    rng = np.random.default_rng(args.seed)

    probed = []
    for ti, task in enumerate(tasks):
        gs = []
        for k in range(args.N):
            prev = existing.get(task["id"], {}).get(k)
            if prev is not None:
                gs.append(prev)
                continue
            t0 = time.time()
            temp = 0.6 + 0.15 * float(rng.random())
            text, _ = gen(
                model,
                tokenizer,
                [{"role": "user", "content": task["prompt"]}],
                args.max_tokens,
                temp=temp,
            )
            g = oracle_G(task, text)
            gs.append(g)
            print(
                f"[{ti+1}/{len(tasks)}] {task['id']} c{k} G={g:.0f} {time.time()-t0:.1f}s",
                flush=True,
            )
            if args.raw_out:
                args.raw_out.parent.mkdir(parents=True, exist_ok=True)
                with args.raw_out.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "task_id": task["id"],
                                "cand_idx": k,
                                "G": g,
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        rate = float(np.mean(gs))
        probed.append({**task, "pass_rate": rate, "round0_G": gs})

    finalize(args, probed)


def finalize(args, probed: list[dict]) -> None:
    """Keep-window filter, tertile bins, per-bin clustering, manifest write.
    Runs after (or instead of, with --from-raw) the expensive probe loop, so
    every failure here must degrade gracefully rather than discard inference."""
    kept = [p for p in probed if args.lo <= p["pass_rate"] <= args.hi]
    kept.sort(key=lambda p: abs(p["pass_rate"] - 0.45))  # prefer mid-band
    kept = kept[: args.keep]

    rates = np.array([p["pass_rate"] for p in kept], dtype=float)
    bins = tertile_bins(rates)
    # cluster within each bin
    labels = [0] * len(kept)
    for bname in ("easy", "mid", "hard"):
        idxs = [i for i, b in enumerate(bins) if b == bname]
        if not idxs:
            continue
        texts = [kept[i].get("question_text") or kept[i]["prompt"] for i in idxs]
        try:
            cl = tfidf_cluster(texts, k=2, seed=args.seed)
        except Exception as e:  # never lose the probe results to clustering
            print(f"WARN: clustering failed ({e!r}); assigning cluster 0", flush=True)
            cl = [0] * len(texts)
        # Power floor: each bin x cluster cell must hold >= 8 items (H3 needs
        # >= 8 items per trajectory). If free clustering is too unbalanced,
        # fall back to the deterministic balanced median split for this bin.
        n0, n1 = cl.count(0), cl.count(1)
        floor = min(8, len(texts) // 2)
        if len(texts) >= 2 and min(n0, n1) < floor:
            print(
                f"WARN: bin '{bname}' clusters unbalanced ({n0}/{n1}, "
                f"floor {floor}); using balanced median split", flush=True,
            )
            cl = balanced_split(texts)
        for j, i in enumerate(idxs):
            labels[i] = cl[j]

    for i, p in enumerate(kept):
        p["difficulty_bin"] = bins[i]
        p["cluster"] = str(labels[i])
        # drop bulky fields not needed at load time? keep gold/test for harness
        p.pop("round0_G", None)

    manifest = {
        "model": args.model,
        "domain": args.domain,
        "seed": args.seed,
        "N_screen": args.N,
        "pass_rate_window": [args.lo, args.hi],
        "pool_size": len(probed),
        "n_in_window": sum(1 for p in probed if args.lo <= p["pass_rate"] <= args.hi),
        "n_kept": len(kept),
        "from_raw": str(args.from_raw) if args.from_raw else None,
        "pass_rate_summary": {
            "mean": float(rates.mean()) if len(rates) else None,
            "min": float(rates.min()) if len(rates) else None,
            "max": float(rates.max()) if len(rates) else None,
        },
        "bin_counts": {
            b: sum(1 for x in bins if x == b) for b in ("easy", "mid", "hard")
        },
        "power_check": power_check(len(kept)),
        "tasks": kept,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in manifest if k != "tasks"}, indent=2))
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
