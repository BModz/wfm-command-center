"""NileConnect WFM Command Centre: entry point. Builds the navigation, grouped by planning stage."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NileConnect WFM", page_icon=":material/support_agent:", layout="wide", initial_sidebar_state="auto")

import ui  # noqa: E402  (importing ui also refreshes the wfm package if its source changed)

sections: dict[str, list] = {}
for page in ui.PAGES:
    sections.setdefault(page.section, []).append(
        st.Page(page.file, title=page.title, icon=page.icon, url_path=page.url_path, default=page.key == "home")
    )
navigation = st.navigation(sections)
ui.install_chrome()
navigation.run()
ui.release()
