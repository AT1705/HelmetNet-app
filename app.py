import streamlit as st

from shared.style import inject_global_css, render_html
from shared.components import navbar, hero_section, stats_row, section_title, feature_grid, cta_section


st.set_page_config(
    page_title="HelmetNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_css()
navbar(active="home")

hero_section()
stats_row()

render_html('<div class="hn-section hn-section-slate">')
section_title(
    "Powerful Features for Road Safety",
    "Advanced technology designed to save lives and improve traffic safety compliance",
    center=True,
)
feature_grid()
render_html("</div>")

cta_section()

