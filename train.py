"""
train.py — train the REAL vs SCREEN classifier.

Usage:
    python train.py --data ./data            # expects data/real/*, data/screen/*
    python train.py --data ./data --group    # device/screen-disjoint validation

Why --group matters (honesty):
    A RANDOM train/val split leaks. Multiple photos of the SAME screen, or several
    patches of the same scene, land in both splits, so val accuracy looks great and
    then collapses on the graders' held-out phones. With --group, whole groups are
    held out. Name files like  <group>__whatever.jpg  (e.g. samsung_oled__01.jpg)
    and the prefix before "__" is treated as the group (a screen, a room, a device).

Outputs:
    model.pkl        the trained pipeline (scaler + classifier)
    features.npz     cached features (delete to recompute)
Prints honest val accuracy, ROC-AUC, and a suggested decision threshold.
"""

import argparse
import os
import sys

import joblib
import numpy as np

from features import FEATURE_NAMES, extract_features

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMG_EXT)
    ]


def group_of(path):
    base = os.path.basename(path)
    return base.split("__")[0] if "__" in base else base  # fallback: per-file


def build_dataset(data_dir):
    reals = list_images(os.path.join(data_dir, "real"))
    screens = list_images(os.path.join(data_dir, "screen"))
    if not reals or not screens:
        sys.exit(
            f"Need images in {data_dir}/real and {data_dir}/screen "
            f"(found {len(reals)} real, {len(screens)} screen)."
        )
    paths = reals + screens
    y = np.array([0] * len(reals) + [1] * len(screens))
    groups = np.array([group_of(p) for p in paths])
    print(f"Extracting features: {len(reals)} real, {len(screens)} screen ...")
    X = np.stack([extract_features(p) for p in paths]).astype(np.float32)
    return X, y, groups, paths


def grouped_split(y, groups, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_groups = set(uniq[:n_val])
    val = np.array([g in val_groups for g in groups])
    return ~val, val


def suggest_threshold(y_true, scores, target_fpr=0.02):
    """Lowest threshold that keeps false-positive rate (real flagged as screen)
    at or below target_fpr on validation. Fraud flagging usually cares more about
    not blocking legit users than about catching every cheater."""
    order = np.argsort(-scores)
    thr_candidates = np.unique(scores)
    best = 0.5
    for t in sorted(thr_candidates):
        pred = scores >= t
        neg = y_true == 0
        fpr = pred[neg].mean() if neg.any() else 0.0
        if fpr <= target_fpr:
            best = float(t)
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument(
        "--group",
        action="store_true",
        help="device/screen-disjoint validation (recommended)",
    )
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--target-fpr", type=float, default=0.02)
    args = ap.parse_args()

    cache = "features.npz"
    if os.path.exists(cache):
        print("Loading cached features (delete features.npz to recompute).")
        d = np.load(cache, allow_pickle=True)
        X, y, groups = d["X"], d["y"], d["groups"]
    else:
        X, y, groups, _ = build_dataset(args.data)
        np.savez(cache, X=X, y=y, groups=groups)

    if args.group and len(set(groups)) >= 4:
        tr, va = grouped_split(y, groups)
        split_kind = (
            f"group-disjoint ({len(set(groups[tr]))} train / "
            f"{len(set(groups[va]))} val groups)"
        )
    else:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        cut = int(len(y) * 0.75)
        tr = np.zeros(len(y), bool)
        tr[idx[:cut]] = True
        va = ~tr
        split_kind = "RANDOM (no groups) — val accuracy is optimistic!"

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Logistic regression on standardised physics features: tiny (a few dozen
    # weights), well-calibrated probabilities, and robust with ~100 samples where
    # tree ensembles overfit. class_weight balances uneven real/screen counts.
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced")),
        ]
    )
    clf.fit(X[tr], y[tr])

    s_va = clf.predict_proba(X[va])[:, 1]
    print(f"\nValidation split: {split_kind}")
    print(f"  n_train={tr.sum()}  n_val={va.sum()}")
    if len(set(y[va])) > 1:
        print(f"  ROC-AUC : {roc_auc_score(y[va], s_va):.4f}")
    thr = suggest_threshold(y[va], s_va, args.target_fpr)
    pred = (s_va >= thr).astype(int)
    print(f"  suggested threshold (<= {args.target_fpr:.0%} FPR): {thr:.3f}")
    print(f"  accuracy @thr: {accuracy_score(y[va], pred):.4f}")
    print(f"  confusion [ [TN FP][FN TP] ]:\n{confusion_matrix(y[va], pred)}")

    # top features by permutation-free proxy: gradient-boosting doesn't expose
    # importances directly here, so just report which raw features separate best.
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    sd = X.std(0) + 1e-6
    sep = np.abs(mu1 - mu0) / sd
    top = np.argsort(-sep)[:8]
    print("\nMost separating features (|Δmean|/std):")
    for i in top:
        print(f"  {FEATURE_NAMES[i]:24s} {sep[i]:.2f}")

    # Refit on ALL data for the shipped model, store threshold.
    clf.fit(X, y)
    joblib.dump(
        {"model": clf, "threshold": thr, "feature_names": FEATURE_NAMES}, args.out
    )
    print(f"\nSaved -> {args.out}  (threshold={thr:.3f})")


if __name__ == "__main__":
    main()
