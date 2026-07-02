"""
export_weights.py — dump the trained model (scaler + logistic) to weights.json
so the browser demo can run the exact same model client-side.

Usage:  python export_weights.py            (reads model.pkl, writes weights.json)
"""

import json

import joblib

bundle = joblib.load("model.pkl")
pipe = bundle["model"]
scaler = pipe.named_steps["scaler"]
lr = pipe.named_steps["lr"]

out = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "coef": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0]),
    "threshold": float(bundle.get("threshold", 0.5)),
    "feature_names": bundle["feature_names"],
}
with open("weights.json", "w") as f:
    json.dump(out, f)
print(
    f"wrote weights.json  ({len(out['coef'])} features, "
    f"threshold={out['threshold']:.3f})"
)
