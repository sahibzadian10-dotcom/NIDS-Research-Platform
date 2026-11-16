"""Post-processing: results.json -> findings, RESULTS.md, figures, dashboard/index.html.

    python -m src.reporting.build_all
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import RESULTS_DIR, ROOT
from src.reporting.charts import make_charts
from src.reporting.reports import make_findings, write_markdown_report


def main():
    p = RESULTS_DIR / "results.json"
    R = json.loads(p.read_text(encoding="utf-8"))
    R["findings"], R["limitations"] = make_findings(R)
    p.write_text(json.dumps(R, separators=(",", ":")), encoding="utf-8")
    write_markdown_report(R, RESULTS_DIR / "RESULTS.md")
    figs = make_charts(R, RESULTS_DIR / "figures")
    print("figures:", figs)
    from dashboard.build_dashboard import build

    out = build(R, ROOT / "dashboard" / "index.html")
    print("dashboard:", out)


if __name__ == "__main__":
    main()
