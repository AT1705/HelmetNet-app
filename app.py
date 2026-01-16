import streamlit as st

from shared.style import inject_global_css
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

st.markdown('<div class="hn-section hn-section-slate">', unsafe_allow_html=True)
section_title(
    "Powerful Features for Road Safety",
    "Advanced technology designed to save lives and improve traffic safety compliance",
    center=True,
)
feature_grid()
st.markdown("</div>", unsafe_allow_html=True)

cta_section()

