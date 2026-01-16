import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HelmetNet",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SHARED UI / BRANDING (HelmetNet look & feel)
# ============================================================
BRAND = {
    "bg": "#F8FAFC",          # slate-50
    "card": "rgba(255,255,255,0.90)",
    "text": "#0F172A",        # slate-900
    "muted": "#475569",       # slate-600
    "border": "rgba(148,163,184,0.35)",  # slate-300 w/ alpha
    "slate700": "#334155",
    "slate800": "#1F2937",
    "slate900": "#0F172A",
    "amber": "#F59E0B",       # amber-500
    "amberHover": "#FBBF24",  # amber-400
}

def inject_global_css(active_page: str) -> None:
    st.markdown(
        f"""
        <style>
          /* App background */
          .stApp {{
            background: {BRAND["bg"]};
            color: {BRAND["text"]};
          }}

          /* Reduce default padding + widen */
          .block-container {{
            padding-top: 5.2rem;   /* room for fixed nav */
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }}

          /* Hide Streamlit chrome */
          #MainMenu, footer, header {{ visibility: hidden; }}
          [data-testid="stStatusWidget"] {{ display: none; }}

          /* --- Top Nav (glass) --- */
          .hn-nav {{
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 9999;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226,232,240,1);
            box-shadow: 0 1px 8px rgba(15,23,42,0.06);
          }}
          .hn-nav-inner {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0.8rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
          }}
          .hn-brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 800;
            color: {BRAND["text"]};
            letter-spacing: -0.02em;
            font-size: 1.15rem;
          }}
          .hn-brand-badge {{
            width: 36px; height: 36px;
            border-radius: 12px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, {BRAND["slate700"]}, {BRAND["slate900"]});
            color: white;
            box-shadow: 0 10px 25px rgba(15,23,42,0.18);
          }}
          .hn-links {{
            display:flex;
            align-items:center;
            gap: 1.1rem;
          }}
          .hn-link {{
            font-weight: 600;
            color: {BRAND["muted"]};
            text-decoration: none;
            padding: 0.35rem 0.2rem;
          }}
          .hn-link:hover {{ color: {BRAND["text"]}; }}
          .hn-link.active {{ color: {BRAND["text"]}; }}

          .hn-cta {{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            padding: 0.55rem 1rem;
            border-radius: 14px;
            font-weight: 800;
            text-decoration:none;
            background: {BRAND["amber"]};
            color: {BRAND["slate900"]};
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

          /* --- Cards / Sections --- */
          .hn-card {{
            background: {BRAND["card"]};
            border: 1px solid {BRAND["border"]};
            border-radius: 18px;
            box-shadow: 0 20px 50px rgba(15,23,42,0.06);
          }}
          .hn-card-pad {{ padding: 1.5rem; }}

          .hn-hero {{
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 30px 80px rgba(15,23,42,0.12);
            background: radial-gradient(1000px 380px at 20% 10%, rgba(245,158,11,0.18), transparent 60%),
                        linear-gradient(135deg, {BRAND["slate800"]}, {BRAND["slate900"]});
            color: white;
          }}
          .hn-hero-inner {{
            padding: 3.2rem 2.2rem;
          }}
          .hn-kicker {{
            display:inline-flex;
            gap: 0.6rem;
            align-items:center;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.78rem;
            opacity: 0.92;
          }}
          .hn-title {{
            margin-top: 0.75rem;
            font-size: 3.2rem;
            line-height: 1.08;
            font-weight: 900;
            letter-spacing: -0.03em;
          }}
          .hn-subtitle {{
            margin-top: 0.9rem;
            font-size: 1.15rem;
            color: rgba(226,232,240,0.92);
            max-width: 52rem;
          }}
          .hn-hero-actions {{
            margin-top: 1.6rem;
            display:flex;
            flex-wrap: wrap;
            gap: 0.8rem;
          }}

          .hn-outline {{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            padding: 0.55rem 1rem;
            border-radius: 14px;
            font-weight: 800;
            text-decoration:none;
            border: 2px solid rgba(255,255,255,0.85);
            color: white;
            background: rgba(255,255,255,0.06);
          }}
          .hn-outline:hover {{
            background: rgba(255,255,255,0.10);
            color: white;
          }}

          .hn-stat {{
            text-align: center;
            padding: 1rem 0.5rem;
          }}
          .hn-stat-val {{
            font-size: 2.2rem;
            font-weight: 900;
            color: {BRAND["slate800"]};
            letter-spacing: -0.02em;
          }}
          .hn-stat-lbl {{
            color: {BRAND["muted"]};
            font-weight: 600;
            margin-top: 0.2rem;
          }}

          .hn-feature {{
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148,163,184,0.35);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 16px 35px rgba(15,23,42,0.05);
            transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
            height: 100%;
          }}
          .hn-feature:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 45px rgba(15,23,42,0.08);
            border-color: rgba(71,85,105,0.55);
          }}
          .hn-feature-icon {{
            width: 44px; height: 44px;
            border-radius: 14px;
            display:flex; align-items:center; justify-content:center;
            background: rgba(241,245,249,1);
            color: {BRAND["slate700"]};
            font-weight: 900;
            margin-bottom: 0.8rem;
          }}
          .hn-feature-title {{
            font-weight: 900;
            color: {BRAND["text"]};
            margin-bottom: 0.3rem;
          }}
          .hn-feature-desc {{
            color: {BRAND["muted"]};
            font-weight: 600;
            font-size: 0.95rem;
            line-height: 1.45;
          }}

          .hn-footer {{
            margin-top: 2.2rem;
            color: {BRAND["muted"]};
            font-weight: 600;
            font-size: 0.9rem;
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

inject_global_css(active_page="home")

# ============================================================
# LANDING CONTENT
# ============================================================
st.markdown(
    """
    <section class="hn-hero">
      <div class="hn-hero-inner">
        <div class="hn-kicker">AI-Powered Road Safety</div>
        <div class="hn-title">Protecting Lives Through Intelligent Helmet Detection</div>
        <div class="hn-subtitle">
          HelmetNet applies computer vision to monitor motorcycle helmet compliance across images,
          videos, and real-time streams—optimized for responsiveness and operational clarity.
        </div>
        <div class="hn-hero-actions">
          <a class="hn-cta" href="/Demo">Try Demo Now</a>
          <a class="hn-outline" href="/About">Learn More</a>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1.1rem'></div>", unsafe_allow_html=True)

# Stats strip
stats = [
    ("99.2%", "Detection Accuracy"),
    ("50K+", "Daily Scans"),
    ("35%", "Reduction in Violations"),
    ("24/7", "Monitoring"),
]
c = st.columns(4)
for i, (val, lbl) in enumerate(stats):
    with c[i]:
        st.markdown(
            f"""
            <div class="hn-card hn-stat">
              <div class="hn-stat-val">{val}</div>
              <div class="hn-stat-lbl">{lbl}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 1.4rem'></div>", unsafe_allow_html=True)

# Feature grid (concise, “SaaS” style)
st.markdown(
    """
    <div class="hn-card hn-card-pad">
      <div style="font-size:1.8rem;font-weight:900;color:#0F172A;letter-spacing:-0.02em;">
        Powerful Features for Road Safety
      </div>
      <div style="margin-top:0.4rem;color:#475569;font-weight:600;">
        Advanced capability wrapped in a clean, compliance-forward experience.
      </div>
      <div style="margin-top:1.2rem;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4, gap="large")
features = [
    ("📸", "Real-Time Detection", "Fast inference with optimized processing for responsive monitoring."),
    ("🛡️", "Enhanced Safety", "Consistent enforcement support with clear violation signaling."),
    ("⚡", "Instant Alerts", "Rate-limited audio alerts when violations are detected."),
    ("📈", "Analytics Ready", "Structured detections for reporting, auditing, and downstream dashboards."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
    with col:
        st.markdown(
            f"""
            <div class="hn-feature">
              <div class="hn-feature-icon">{icon}</div>
              <div class="hn-feature-title">{title}</div>
              <div class="hn-feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="hn-footer">
      HelmetNet | Streamlit UI aligned to the provided HelmetNet reference design.
    </div>
    """,
    unsafe_allow_html=True,
)
