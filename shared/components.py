from __future__ import annotations

import streamlit as st


def navbar(active: str = "home") -> None:
    """Top navigation bar matching the Figma prototype.

    Parameters
    ----------
    active:
        One of: home | about | demo
    """

    active = (active or "home").lower()

    # Fixed navbar (HTML) with Streamlit multipage URLs.
    # Streamlit typically maps pages to /About and /Demo by title.
    st.markdown(
        f"""
<div class="hn-nav">
  <div class="hn-container hn-nav-inner">
    <a class="hn-brand" href="/">
      <span class="hn-brand-icon" aria-hidden="true">
        <!-- minimal shield icon (inline svg) -->
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2l7 4v6c0 5-3 9-7 10-4-1-7-5-7-10V6l7-4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      </span>
      <span>HelmetNet</span>
    </a>

    <div class="hn-nav-links">
      <a class="hn-nav-link {'is-active' if active=='home' else ''}" href="/">Home</a>
      <a class="hn-nav-link {'is-active' if active=='about' else ''}" href="/About">About</a>
      <a class="hn-nav-cta" href="/Demo">Start Demo</a>
    </div>
  </div>
</div>
<div class="hn-nav-spacer"></div>
""",
        unsafe_allow_html=True,
    )


def hero_section() -> None:
    """Landing hero section (Home page) closely matching the React/Tailwind layout."""

    st.markdown(
        """
<section class="hn-hero">
  <div class="hn-hero-bg"></div>
  <div class="hn-hero-overlay"></div>
  <div class="hn-container hn-hero-content">
    <div class="hn-hero-badge">
      <span class="hn-hero-badge-icon" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2l7 4v6c0 5-3 9-7 10-4-1-7-5-7-10V6l7-4z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      </span>
      <span>AI-Powered Road Safety</span>
    </div>

    <h1 class="hn-hero-title">Protecting Lives Through Intelligent Helmet Detection</h1>
    <p class="hn-hero-subtitle">
      HelmetNet uses cutting-edge artificial intelligence to automatically detect and monitor motorcycle helmet compliance,
      making roads safer for everyone.
    </p>

    <div class="hn-hero-actions">
      <a class="hn-btn hn-btn-primary" href="/Demo">Try Demo Now</a>
      <a class="hn-btn hn-btn-outline" href="/About">Learn More</a>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def stats_row() -> None:
    stats = [
        ("99.2%", "Detection Accuracy"),
        ("50K+", "Daily Scans"),
        ("35%", "Reduction in Violations"),
        ("24/7", "Monitoring"),
    ]

    cards = "".join(
        [
            f"""
            <div class="hn-stat">
              <div class="hn-stat-value">{value}</div>
              <div class="hn-stat-label">{label}</div>
            </div>
            """
            for value, label in stats
        ]
    )

    st.markdown(
        f"""
<section class="hn-section hn-section-white">
  <div class="hn-container">
    <div class="hn-stats-grid">
      {cards}
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None, center: bool = True) -> None:
    align = "center" if center else "left"
    subtitle_html = f"<p class='hn-section-subtitle'>{subtitle}</p>" if subtitle else ""

    st.markdown(
        f"""
<div class="hn-container" style="text-align:{align};">
  <h2 class="hn-section-title">{title}</h2>
  {subtitle_html}
</div>
""",
        unsafe_allow_html=True,
    )


def feature_grid() -> None:
    features = [
        (
            "Real-Time Detection",
            "Advanced AI algorithms detect helmet compliance in milliseconds with 99.2% accuracy.",
            "camera",
        ),
        (
            "Enhanced Safety",
            "Automated monitoring ensures consistent enforcement of helmet safety regulations.",
            "shield",
        ),
        (
            "Instant Alerts",
            "Immediate notifications to traffic authorities when violations are detected.",
            "zap",
        ),
        (
            "Analytics Dashboard",
            "Comprehensive insights into compliance rates and traffic patterns over time.",
            "trend",
        ),
    ]

    icon_svg = {
        "camera": "<path d='M20 5h-3.2l-1.6-2H8.8L7.2 5H4a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Z' stroke='currentColor' stroke-width='2' fill='none'/> <circle cx='12' cy='13' r='3.5' stroke='currentColor' stroke-width='2' fill='none'/>",
        "shield": "<path d='M12 2l7 4v6c0 5-3 9-7 10-4-1-7-5-7-10V6l7-4z' stroke='currentColor' stroke-width='2' fill='none' stroke-linejoin='round'/>",
        "zap": "<path d='M13 2L3 14h7l-1 8 12-14h-7l-1-6Z' stroke='currentColor' stroke-width='2' fill='none' stroke-linejoin='round'/>",
        "trend": "<path d='M3 17l6-6 4 4 7-7' stroke='currentColor' stroke-width='2' fill='none' stroke-linecap='round' stroke-linejoin='round'/> <path d='M14 8h7v7' stroke='currentColor' stroke-width='2' fill='none' stroke-linecap='round' stroke-linejoin='round'/>",
    }

    tiles = "".join(
        [
            f"""
            <div class="hn-feature">
              <div class="hn-feature-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">{icon_svg[key]}</svg>
              </div>
              <div class="hn-feature-title">{title}</div>
              <div class="hn-feature-desc">{desc}</div>
            </div>
            """
            for title, desc, key in features
        ]
    )

    st.markdown(
        f"""
<section class="hn-section hn-section-slate">
  <div class="hn-container">
    <div class="hn-features-grid">
      {tiles}
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def cta_section() -> None:
    st.markdown(
        """
<section class="hn-cta">
  <div class="hn-container hn-cta-inner">
    <h2 class="hn-cta-title">Ready to See It in Action?</h2>
    <p class="hn-cta-subtitle">Experience the power of AI-driven helmet detection with our interactive demo</p>
    <a class="hn-btn hn-btn-primary hn-btn-lg" href="/Demo">Launch Demo</a>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def html_table(rows: list[dict]) -> str:
    """Render a styled HTML table matching the Detection Results table."""

    if not rows:
        return (
            "<div class='hn-empty'>No results yet. Upload an image and click \"Run detection\".</div>"
        )

    def esc(x: object) -> str:
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    header = """
      <thead>
        <tr>
          <th>#</th>
          <th>LABEL</th>
          <th>CONFIDENCE</th>
          <th>COMPLIANCE</th>
          <th>BBOX (X,Y,W,H)</th>
        </tr>
      </thead>
    """

    body_rows = []
    for r in rows:
        compliance = esc(r.get("compliance", ""))
        if compliance == "COMPLIANT":
            compliance_html = "<span class='hn-pill hn-pill-green'>COMPLIANT</span>"
        elif compliance in ("", "N/A", None):
            compliance_html = "<span class='hn-muted'>N/A</span>"
        else:
            compliance_html = f"<span class='hn-muted'>{compliance}</span>"

        body_rows.append(
            f"""
            <tr>
              <td class='hn-muted'>{esc(r.get('id',''))}</td>
              <td class='hn-strong'>{esc(r.get('label',''))}</td>
              <td><span class='hn-green'>{esc(r.get('confidence',''))}%</span></td>
              <td>{compliance_html}</td>
              <td class='hn-mono'>{esc(r.get('bbox',''))}</td>
            </tr>
            """
        )

    table = (
        "<div class='hn-table-wrap'><table class='hn-table'>"
        + header
        + "<tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )
    return table
