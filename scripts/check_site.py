"""Check local links and publish boundaries without dependencies."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / "_site"


class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                url = urlsplit(value)
                if url.scheme or url.netloc or not url.path:
                    continue
                assert not url.path.startswith("/"), f"Root-relative URL breaks /yuri/: {value}"
                target = (ROOT / url.path).resolve()
                assert target.is_relative_to(ROOT.resolve()), value
                assert target.exists(), f"Missing local resource: {value}"


for page in ("index.html", "docs.html"):
    source = (ROOT / page).read_text(encoding="utf-8")
    assert 'lang="zh-CN"' in source and 'name="viewport"' in source
    Links().feed(source)
expected = {
    "index.html", "docs.html", "style.css", "app.js", "policy.mjs", "favicon.svg",
    "development.md", ".nojekyll", "assets/sc2-battlefield.png",
}
actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
assert actual == expected, "Unexpected files in publish artifact"
assert 'url("./assets/sc2-battlefield.png")' in (ROOT / "style.css").read_text(encoding="utf-8")
print("Static links, /yuri/ compatibility and artifact allowlist passed")
