import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="About | HelmetNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SHARED UI / BRANDING (match app.py exactly)
# ============================================================
BRAND = {
    "bg": "#F8FAFC",
    "card": "rgba(255,255,255,0.90)",
    "text": "#0F172A",
    "muted": "#475569",
    "border": "rgba(148,163,184,0.35)",
    "slate700": "#334155",
    "slate800": "#1F2937",
    "slate900": "#0F172A",
    "amber": "#F59E0B",
    "amberHover": "#FBBF24",
}

def inject_global_css(active_page: str) -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {BRAND["bg"]}; color: {BRAND["text"]}; }}
          .block-container {{
            padding-top: 5.2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }}
          #MainMenu, footer, header {{ visibility: hidden; }}
          [data-testid="stStatusWidget"] {{ display: none; }}

          .hn-nav {{
            position: fixed; top: 0; left: 0; right: 0;
            z-index: 9999;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226,232,240,1);
            box-shadow: 0 1px 8px rgba(15,23,42,0.06);
          }}
          .hn-nav-inner {{
            max-width: 1200px; margin: 0 auto;
            padding: 0.8rem 1rem;
            display:flex; align-items:center; justify-content:space-between;
          }}
          .hn-brand {{
            display:flex; align-items:center; gap:0.6rem;
            font-weight: 800; font-size: 1.15rem; letter-spacing:-0.02em;
            color: {BRAND["text"]};
          }}
          .hn-brand-badge {{
            width: 36px; height: 36px; border-radius: 12px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, {BRAND["slate700"]}, {BRAND["slate900"]});
            color:white; box-shadow: 0 10px 25px rgba(15,23,42,0.18);
          }}
          .hn-links {{ display:flex; align-items:center; gap:1.1rem; }}
          .hn-link {{
            font-weight: 600; color: {BRAND["muted"]};
            text-decoration:none; padding: 0.35rem 0.2rem;
          }}
          .hn-link:hover {{ color: {BRAND["text"]}; }}
          .hn-link.active {{ color: {BRAND["text"]}; }}

          .hn-cta {{
            display:inline-flex; align-items:center; justify-content:center;
            padding: 0.55rem 1rem; border-radius: 14px;
            font-weight: 800; text-decoration:none;
            background: {BRAND["amber"]}; color: {BRAND["slate900"]};
            box-shadow: 0 10px 25px rgba(245,158,11,0.25);
            border: 1px solid rgba(245,158,11,0.35);
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
          }}
          .hn-cta:hover {{
            background: {BRAND["amberHover"]};
            transform: translateY(-1px);
            box-shadow: 0 14px 35px rgba(245,158,11,0.28);
            color: {BRAND["slate900"]};
          }}

          .hn-hero {{
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 30px 80px rgba(15,23,42,0.12);
            background: linear-gradient(135deg, {BRAND["slate800"]}, {BRAND["slate900"]});
            color: white;
          }}
          .hn-hero-inner {{ padding: 2.7rem 2.2rem; }}
          .hn-title {{
            font-size: 2.6rem; line-height: 1.1;
            font-weight: 900; letter-spacing:-0.03em;
            margin: 0;
          }}
          .hn-sub {{
            margin-top: 0.6rem;
            color: rgba(226,232,240,0.92);
            font-size: 1.05rem;
            max-width: 58rem;
            font-weight: 600;
          }}

          .hn-card {{
            background: {BRAND["card"]};
            border: 1px solid {BRAND["border"]};
            border-radius: 18px;
            box-shadow: 0 20px 50px rgba(15,23,42,0.06);
          }}
          .hn-card-pad {{ padding: 1.5rem; }}

          .hn-h2 {{
            font-size: 1.7rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            color: {BRAND["text"]};
            margin-bottom: 0.8rem;
          }}
          .hn-p {{
            color: {BRAND["muted"]};
            font-weight: 600;
            line-height: 1.55;
            font-size: 1.02rem;
          }}

          .hn-exp {{
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148,163,184,0.35);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 16px 35px rgba(15,23,42,0.05);
            height: 100%;
          }}
          .hn-exp-badge {{
            width: 46px; height: 46px;
            border-radius: 14px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, {BRAND["slate700"]}, {BRAND["slate800"]});
            color:white;
            font-weight: 900;
            box-shadow: 0 12px 28px rgba(15,23,42,0.18);
            flex-shrink: 0;
          }}
          .hn-exp-title {{
            font-weight: 900;
            color: {BRAND["text"]};
            font-size: 1.05rem;
            margin-top: 0.1rem;
          }}
          .hn-exp-k {{
            font-weight: 900;
            color: {BRAND["slate700"]};
            margin-top: 0.9rem;
            font-size: 0.95rem;
          }}
          .hn-exp-v {{
            color: {BRAND["muted"]};
            font-weight: 600;
            margin-top: 0.2rem;
            line-height: 1.45;
          }}
        </style>

        <div class="hn-nav">
          <div class="hn-nav-inner">
            <div class="hn-brand">
              <div class="hn-brand-badge">🛡️</div>
              HelmetNet
            </div>
            <div class="hn-links">
              <a class="hn-link {"active" if active_page=="home" else ""}" href="/">Home</a>
              <a class="hn-link {"active" if active_page=="about" else ""}" href="/About">About</a>
              <a class="hn-cta" href="/Demo">Start Demo</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_global_css(active_page="about")

# ============================================================
# ABOUT PAGE CONTENT
# ============================================================
st.markdown(
    """
    <section class="hn-hero">
      <div class="hn-hero-inner">
        <h1 class="hn-title">About HelmetNet</h1>
        <div class="hn-sub">
          A computer vision pipeline designed to detect helmet compliance for motorcycle riders through iterative research and development.
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)

# --- Project Overview (single styled container as requested) ---
st.markdown(
    """
    <div class="hn-card hn-card-pad">
      <div class="hn-h2">About HelmetNet</div>
      <div class="hn-p">
        HelmetNet focuses on operational helmet compliance detection across images, videos, and real-time streams.
        The project emphasizes iterative improvements driven by dataset labeling discipline, class definition clarity,
        and deployment-oriented performance constraints.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)

# --- The 4 Experiments (2x2 grid) ---
st.markdown(
    """
    <div style="text-align:center; margin: 0.2rem 0 1rem 0;">
      <div style="font-size:1.9rem;font-weight:900;letter-spacing:-0.02em;color:#0F172A;">The 4 Experiments</div>
      <div style="color:#475569;font-weight:600;">Research progression addressing concrete failure modes and refining the dataset</div>
    </div>
    """,
    unsafe_allow_html=True,
)

experiments = [
    {
        "number": 1,
        "title": "Poor cap detection (baseline limitations)",
        "issue": "The model frequently confused caps / head coverings with helmets, producing poor discrimination.",
        "learning": "Model performance is bottlenecked by labeling policy quality more than raw architecture.",
    },
    {
        "number": 2,
        "title": "Helmet type classification refinement",
        "issue": "System struggled to distinguish between different helmet types and safety standards.",
        "learning": "Dataset diversity across helmet types and viewing angles is critical for robust detection.",
    },
    {
        "number": 3,
        "title": "Real-time stream optimization",
        "issue": "Processing latency exceeded acceptable thresholds for live traffic monitoring.",
        "learning": "Architecture optimization and edge computing integration essential for real-time applications.",
    },
    {
        "number": 4,
        "title": "Multi-angle detection enhancement",
        "issue": "Detection accuracy dropped significantly for side and rear viewing angles.",
        "learning": "Comprehensive multi-angle dataset coverage ensures consistent performance across deployment scenarios.",
    },
]

grid = st.columns(2, gap="large")
for idx, exp in enumerate(experiments):
    with grid[idx % 2]:
        st.markdown(
            f"""
            <div class="hn-exp">
              <div style="display:flex; gap:0.9rem; align-items:flex-start;">
                <div class="hn-exp-badge">E{exp["number"]}</div>
                <div>
                  <div class="hn-exp-title">{exp["title"]}</div>
                </div>
              </div>

              <div class="hn-exp-k">Issue</div>
              <div class="hn-exp-v">{exp["issue"]}</div>

              <div class="hn-exp-k">Learning</div>
              <div class="hn-exp-v">{exp["learning"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
