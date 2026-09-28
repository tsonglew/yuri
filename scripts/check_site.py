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
expected = {"index.html", "docs.html", "style.css", "app.js", "policy.mjs", "favicon.svg", "development.md", ".nojekyll"}
assert {p.name for p in ROOT.iterdir()} == expected, "Unexpected files in publish artifact"
print("Static links, /yuri/ compatibility and artifact allowlist passed")
