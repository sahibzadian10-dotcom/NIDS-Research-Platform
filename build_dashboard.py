"""Build the self-contained research dashboard (one HTML file, no server needed).

    python -m dashboard.build_dashboard            # results.json -> dashboard/index.html
    python -m dashboard.build_dashboard --fragment # same page without <html>/<body> wrappers

The template (dashboard/template.html) holds all markup, styles and chart code; this script only embeds
experiments/results/results.json into it. Open dashboard/index.html in any browser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import RESULTS_DIR, ROOT

TEMPLATE = Path(__file__).with_name("template.html")
TOKEN = "/*__RESULTS__*/null"
WRAP_HEAD = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
)


def build(results: dict, out_path, fragment: bool = False) -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    if TOKEN not in html:
        raise RuntimeError("template.html is missing the results placeholder")
    payload = json.dumps(results, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = html.replace(TOKEN, payload)
    if not fragment:
        html = WRAP_HEAD + html + "\n</body>\n</html>\n"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DIR / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "index.html"))
    ap.add_argument("--fragment", action="store_true")
    args = ap.parse_args(argv)
    R = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = build(R, args.out, fragment=args.fragment)
    print(f"dashboard: {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
