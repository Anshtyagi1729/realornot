# organize.py — run: uv run python organize.py
import pathlib
import shutil

src, dst = pathlib.Path("raw"), pathlib.Path("data")
for cls in ("real", "screen"):
    (dst / cls).mkdir(parents=True, exist_ok=True)
    for group_dir in (src / cls).iterdir():
        if not group_dir.is_dir():
            continue
        group = group_dir.name  # folder name = group
        imgs = [
            p
            for p in sorted(group_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
        ]
        for i, p in enumerate(imgs, 1):
            shutil.copy(p, dst / cls / f"{group}__{i:02d}{p.suffix.lower()}")
    print(cls, "done")
