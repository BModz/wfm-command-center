"""Smoke test: every app page renders through the real navigation without raising."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from ui import PAGES  # noqa: E402


@pytest.mark.parametrize("page", PAGES, ids=[p.key for p in PAGES])
def test_page_renders(page):
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=120)
    at.run()
    if page.key != "home":
        at.switch_page(page.file)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    heading = "<h1>WFM Command Centre</h1>" if page.key == "home" else f"<h1>{page.title}</h1>"
    assert any(heading in md.value for md in at.markdown), f"{page.title} heading not rendered"
