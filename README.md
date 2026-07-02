# Spot the Fake Photo — REAL vs PHOTO-OF-A-SCREEN

Given one image, outputs a score in [0,1] (0 = real photo, 1 = photo of a
screen), using hand-crafted FFT/residual/colour features and a logistic
regression classifier (`train.py` trains it, `predict.py image.jpg` scores
one image, `model.pkl` is the trained model). `features.py` and `model.pkl`
must sit next to `predict.py` to run. `fe/` is a live browser demo that runs
the same model fully client-side (`fe/serve.py` to run it locally, or see
the GitHub Pages deploy workflow).
