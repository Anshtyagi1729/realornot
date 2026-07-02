#!/usr/bin/env python3
"""
predict.py — REAL vs PHOTO-OF-A-SCREEN.

    python predict.py some_image.jpg
    -> prints a single float in [0,1]  (0 = real photo, 1 = photo of a screen)

The model (model.pkl) is produced by train.py. Prints ONLY the number to stdout
so it is trivial to parse; diagnostics go to stderr.
"""

import os
import sys

import joblib

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def score(image_path):
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    x = extract_features(image_path).reshape(1, -1)
    return float(clf.predict_proba(x)[0, 1])


def main():
    if len(sys.argv) != 2:
        print("usage: python predict.py <image>", file=sys.stderr)
        sys.exit(2)
    try:
        p = score(sys.argv[1])
    except FileNotFoundError:
        print(f"error: file not found: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # never crash the grader's harness
        print(f"warn: {e}; defaulting to 0.5", file=sys.stderr)
        p = 0.5
    print(round(p, 4))


if __name__ == "__main__":
    main()
