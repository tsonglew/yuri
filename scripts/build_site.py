"""Build an allowlisted static artifact; never copy the repository wholesale."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "_site"
BACKGROUND_ASSETS = ("sc2-battlefield.png",)


def main():
    OUTPUT.mkdir(exist_ok=True)
    for name in ("index.html", "docs.html", "style.css", "app.js", "policy.mjs", "favicon.svg"):
        shutil.copyfile(ROOT / "site" / name, OUTPUT / name)
    asset_dir = OUTPUT / "assets"
    asset_dir.mkdir(exist_ok=True)
    for name in BACKGROUND_ASSETS:
        shutil.copyfile(ROOT / "site" / "assets" / name, asset_dir / name)
    shutil.copyfile(ROOT / "docs/DEVELOPMENT.md", OUTPUT / "development.md")
    (OUTPUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built static site: {OUTPUT}")


if __name__ == "__main__":
    main()
