"""Capture full-page screenshots of every app page with headless Chromium.

Start the app first (``streamlit run app/Home.py``), then run:

    python scripts/take_screenshots.py [base_url]

Set CHROMIUM_PATH to use an existing Chromium binary.

Fails if any page shows a Streamlit exception.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {
    "home": "",
    "overview": "Overview",
    "forecast": "Forecast",
    "staffing": "Staffing",
    "capacity_plan": "Capacity_Plan",
    "schedule": "Schedule",
    "rta_desk": "Real_Time_RTA_Desk",
}
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main(base_url: str = "http://localhost:8501") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    with sync_playwright() as p:
        # CHROMIUM_PATH lets you point at a preinstalled browser instead of running `playwright install`.
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        for name, path in PAGES.items():
            page.goto(f"{base_url}/{path}", wait_until="networkidle")
            page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60_000)
            page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000
            )
            page.wait_for_timeout(2500)  # let Plotly finish drawing
            if page.locator('[data-testid="stException"]').count():
                failures.append(name)
            # Streamlit scrolls inside its main container, so grow the viewport to the content height.
            height = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
            page.set_viewport_size({"width": 1440, "height": int(height) + 40})
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            page.set_viewport_size({"width": 1440, "height": 900})
            print(f"captured {name}")
        browser.close()
    if failures:
        raise SystemExit(f"Pages with errors: {', '.join(failures)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
