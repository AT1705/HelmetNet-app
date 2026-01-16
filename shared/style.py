import streamlit as st


def inject_global_css() -> None:
    """Inject HelmetNet global CSS to replicate the Figma/Tailwind look-and-feel."""

    st.markdown(
        """
<style>
/* =====================
   HelmetNet Design Tokens (extracted / inferred)
   ===================== */
:root{
  /* Palette (Tailwind-like) */
  --hn-slate-50: #f8fafc;
  --hn-slate-100: #f1f5f9;
  --hn-slate-200: #e2e8f0;
  --hn-slate-300: #cbd5e1;
  --hn-slate-400: #94a3b8;
  --hn-slate-500: #64748b;
  --hn-slate-600: #475569;
  --hn-slate-700: #334155;
  --hn-slate-800: #1e293b;
  --hn-slate-900: #0f172a;

  --hn-amber-50: #fffbeb;
  --hn-amber-200: #fde68a;
  --hn-amber-300: #fcd34d;
  --hn-amber-400: #fbbf24;
  --hn-amber-500: #f59e0b;

  --hn-green-500: #22c55e;

  /* Core tokens (from theme.css + Figma usage) */
  --hn-bg: var(--hn-slate-50);
  --hn-surface: #ffffff;
  --hn-text: var(--hn-slate-900);
  --hn-text-muted: var(--hn-slate-600);
  --hn-border: var(--hn-slate-200);
  --hn-focus: var(--hn-amber-500);

  /* Typography */
  --hn-font: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";

  /* Radius & shadows */
  --hn-radius-lg: 12px;  /* rounded-xl */
  --hn-radius-md: 10px;  /* theme radius ~0.625rem */
  --hn-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.08);
  --hn-shadow-md: 0 10px 15px -3px rgba(15, 23, 42, 0.12), 0 4px 6px -4px rgba(15, 23, 42, 0.12);
  --hn-shadow-lg: 0 20px 25px -5px rgba(15, 23, 42, 0.18), 0 8px 10px -6px rgba(15, 23, 42, 0.18);
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* =====================
   Streamlit chrome removal
   ===================== */
header[data-testid="stHeader"],
footer,
#MainMenu {
  display: none !important;
}

/* Remove top padding caused by hidden header */
section.main > div {
  padding-top: 0rem;
}

/* Hide multipage sidebar navigation (keep sidebar content for Demo settings) */
div[data-testid="stSidebarNav"] {
  display: none !important;
}

/* =====================
   Global layout
   ===================== */
html, body, [class*="stApp"] {
  font-family: var(--hn-font) !important;
  background: var(--hn-bg) !important;
  color: var(--hn-text) !important;
}

/* Full-width canvas */
.main .block-container {
  max-width: 100% !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 2rem !important;
}

/* Streamlit default elements cleanup */
div[data-testid="stDecoration"],
div[data-testid="stToolbar"] {
  display: none !important;
}

/* =====================
   Reusable primitives
   ===================== */
.hn-container{
  max-width: 80rem; /* max-w-7xl */
  margin: 0 auto;
  padding: 0 1.5rem;
}

.hn-surface{
  background: var(--hn-surface);
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  box-shadow: var(--hn-shadow-md);
}

.hn-card{
  background: var(--hn-surface);
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  box-shadow: var(--hn-shadow-md);
}

.hn-muted{
  color: var(--hn-text-muted);
}

/* Headings (Streamlit markdown) */
h1, h2, h3 {
  color: var(--hn-text) !important;
  letter-spacing: -0.01em;
}

/* =====================
   Buttons & inputs
   ===================== */
/* Streamlit buttons */
.stButton > button {
  border-radius: var(--hn-radius-lg) !important;
  padding: 0.9rem 1.25rem !important;
  border: 1px solid transparent !important;
  font-weight: 700 !important;
  background: var(--hn-amber-500) !important;
  color: var(--hn-slate-900) !important;
  box-shadow: var(--hn-shadow-md) !important;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

.stButton > button:hover {
  background: var(--hn-amber-400) !important;
  box-shadow: var(--hn-shadow-lg) !important;
  transform: translateY(-1px);
}

.stButton > button:active {
  transform: translateY(0);
}

/* Secondary button helper */
.hn-btn-secondary .stButton > button {
  background: transparent !important;
  color: #ffffff !important;
  border: 2px solid rgba(255,255,255,0.9) !important;
  box-shadow: none !important;
}

.hn-btn-secondary .stButton > button:hover {
  background: rgba(255,255,255,0.10) !important;
  box-shadow: none !important;
  transform: none;
}

/* Selectbox / text inputs */
[data-testid="stSelectbox"],
[data-testid="stTextInput"],
[data-testid="stNumberInput"],
[data-testid="stSlider"] {
  font-family: var(--hn-font) !important;
}

/* Selectbox container */
div[data-testid="stSelectbox"] > div {
  border-radius: 10px !important;
}

/* Slider accent (best-effort across versions) */
div[data-testid="stSlider"] [role="slider"] {
  outline-color: var(--hn-focus) !important;
}

/* File uploader styling */
section[data-testid="stFileUploader"] {
  border: 2px dashed var(--hn-slate-300) !important;
  border-radius: var(--hn-radius-lg) !important;
  padding: 1.25rem !important;
  background: rgba(255,255,255,0.75);
}

section[data-testid="stFileUploader"]:hover{
  border-color: var(--hn-amber-500) !important;
  background: rgba(255, 251, 235, 0.35);
}


/* =====================
   Radio as tabs (Demo mode selector)
   ===================== */
div[data-testid="stRadio"] div[role="radiogroup"]{
  background: #ffffff !important;
  border: 1px solid var(--hn-border) !important;
  border-radius: var(--hn-radius-lg) !important;
  padding: 6px !important;
  box-shadow: var(--hn-shadow-md) !important;
  gap: 8px !important;
}

div[data-testid="stRadio"] div[data-baseweb="radio"]{
  border-radius: 10px !important;
  padding: 10px 12px !important;
  margin: 0 !important;
  flex: 1 1 0 !important;
  justify-content: center !important;
  background: transparent !important;
}

div[data-testid="stRadio"] div[data-baseweb="radio"][aria-checked="true"]{
  background: var(--hn-amber-500) !important;
  box-shadow: var(--hn-shadow-md) !important;
}

div[data-testid="stRadio"] div[data-baseweb="radio"] *{
  color: var(--hn-slate-600) !important;
  font-weight: 700 !important;
}

div[data-testid="stRadio"] div[data-baseweb="radio"][aria-checked="true"] *{
  color: var(--hn-slate-900) !important;
}

/* Hide native radio icon */
div[data-testid="stRadio"] svg{ display:none !important; }

/* =====================
   Sidebar (Demo configuration)
   ===================== */
section[data-testid="stSidebar"] {
  background: var(--hn-surface) !important;
  border-right: 1px solid var(--hn-border) !important;
}

section[data-testid="stSidebar"] .block-container{
  padding-top: 1rem !important;
}

/* =====================
   HTML Buttons / Links
   ===================== */
.hn-btn{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding: 16px 32px;
  border-radius: var(--hn-radius-lg);
  font-weight: 800;
  text-decoration:none;
  transition: background 120ms ease, box-shadow 120ms ease, transform 120ms ease, border-color 120ms ease;
}

.hn-btn-primary{
  background: var(--hn-amber-500);
  color: var(--hn-slate-900);
  box-shadow: var(--hn-shadow-lg);
}

.hn-btn-primary:hover{
  background: var(--hn-amber-400);
  transform: translateY(-1px);
}

.hn-btn-outline{
  background: transparent;
  color: #ffffff;
  border: 2px solid rgba(255,255,255,0.9);
  box-shadow: none;
}

.hn-btn-outline:hover{
  background: rgba(255,255,255,0.10);
}

.hn-btn-lg{ padding: 16px 40px; font-size: 1.05rem; }

/* =====================
   Hero (Landing)
   ===================== */
.hn-hero{
  position: relative;
  height: 90vh;
  min-height: 600px;
  display:flex;
  align-items:center;
  color: #ffffff;
  overflow:hidden;
}

.hn-hero-bg{
  position:absolute;
  inset:0;
  background-image: url('https://images.unsplash.com/photo-1645094118521-5c72e98985e5?auto=format&fit=crop&w=1920&q=80');
  background-size: cover;
  background-position: center;
  opacity: 0.20;
  transform: scale(1.02);
}

.hn-hero-overlay{
  position:absolute;
  inset:0;
  background: linear-gradient(135deg, var(--hn-slate-800) 0%, var(--hn-slate-900) 100%);
}

.hn-hero-content{
  position: relative;
  z-index: 2;
  max-width: 48rem;
}

.hn-hero-badge{
  display:inline-flex;
  align-items:center;
  gap: 10px;
  margin-bottom: 1.5rem;
  font-size: 0.85rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 700;
  color: rgba(255,255,255,0.92);
}

.hn-hero-title{
  font-size: clamp(2.2rem, 4vw, 3.75rem);
  line-height: 1.05;
  font-weight: 900;
  margin: 0 0 1.25rem 0;
}

.hn-hero-subtitle{
  font-size: 1.25rem;
  line-height: 1.6;
  margin: 0 0 2rem 0;
  color: rgba(226,232,240,0.95);
}

.hn-hero-actions{ display:flex; gap: 16px; flex-wrap: wrap; }

/* =====================
   Sections
   ===================== */
.hn-section{ padding: 80px 0; }
.hn-section-white{ background: #ffffff; }
.hn-section-slate{ background: var(--hn-slate-50); }

.hn-section-title{
  font-size: 2.25rem;
  font-weight: 900;
  margin: 0 0 0.75rem 0;
}

.hn-section-subtitle{
  margin: 0 auto;
  max-width: 60rem;
  font-size: 1.25rem;
  color: #4b5563;
}

/* Stats */
.hn-stats-grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 32px;
}

@media (max-width: 900px){
  .hn-stats-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}

.hn-stat{ text-align:center; }
.hn-stat-value{ font-size: 2.25rem; font-weight: 900; color: var(--hn-slate-800); margin-bottom: 6px; }
.hn-stat-label{ color: #4b5563; }

/* Features */
.hn-features-grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 24px;
}

@media (max-width: 1100px){
  .hn-features-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); }
}

@media (max-width: 640px){
  .hn-features-grid{ grid-template-columns: repeat(1, minmax(0,1fr)); }
}

.hn-feature{
  background: #ffffff;
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  padding: 24px;
  transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
}

.hn-feature:hover{
  border-color: var(--hn-slate-400);
  box-shadow: var(--hn-shadow-lg);
  transform: translateY(-2px);
}

.hn-feature-icon{
  width: 48px;
  height: 48px;
  border-radius: var(--hn-radius-lg);
  background: var(--hn-slate-100);
  display:flex;
  align-items:center;
  justify-content:center;
  margin-bottom: 16px;
  color: var(--hn-slate-700);
}

.hn-feature-title{
  font-size: 1.25rem;
  font-weight: 800;
  margin-bottom: 8px;
  color: var(--hn-slate-900);
}

.hn-feature-desc{ color: #4b5563; font-size: 0.95rem; line-height: 1.55; }

/* CTA */
.hn-cta{
  padding: 80px 0;
  background: linear-gradient(135deg, var(--hn-slate-800) 0%, var(--hn-slate-900) 100%);
  color: #ffffff;
}

.hn-cta-inner{ text-align:center; max-width: 56rem; }
.hn-cta-title{ font-size: 2.25rem; font-weight: 900; margin: 0 0 0.75rem 0; }
.hn-cta-subtitle{ font-size: 1.25rem; color: rgba(226,232,240,0.85); margin: 0 0 2rem 0; }

/* =====================
   Small helpers used by Demo
   ===================== */
.hn-empty{
  text-align:center;
  padding: 48px 24px;
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  background: var(--hn-slate-50);
  color: var(--hn-slate-500);
}

.hn-strong{ font-weight: 700; color: var(--hn-slate-900); }
.hn-green{ color: var(--hn-green-500); font-weight: 800; }

.hn-pill-green{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  color: var(--hn-green-500);
  font-weight: 800;
}

/* =====================
   About page layouts
   ===================== */
.hn-about-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 32px;
  align-items: start;
}
@media (max-width: 980px){
  .hn-about-grid{ grid-template-columns: 1fr; }
}

.hn-about-p{
  margin: 0 0 16px 0;
  font-size: 1.05rem;
  color: #374151;
  line-height: 1.75;
}

.hn-about-callout{
  margin-top: 20px;
  background: var(--hn-slate-50);
  border-left: 4px solid var(--hn-amber-500);
  border-radius: var(--hn-radius-lg);
  padding: 18px 18px;
  color: #374151;
  line-height: 1.6;
}

.hn-about-media{ display:flex; }
.hn-about-image{
  width: 100%;
  height: 350px;
  border-radius: var(--hn-radius-lg);
  border: 1px solid var(--hn-border);
  box-shadow: var(--hn-shadow-md);
  background-image: url('https://images.unsplash.com/photo-1569932353341-b518d82f8a54?auto=format&fit=crop&w=1600&q=80');
  background-size: cover;
  background-position: center;
}

.hn-ref-grid{
  display:grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 18px;
}
@media (max-width: 900px){
  .hn-ref-grid{ grid-template-columns: 1fr; }
}

.hn-ref{
  background: var(--hn-slate-50);
  border-left: 4px solid var(--hn-slate-700);
  border-radius: var(--hn-radius-lg);
  padding: 18px;
}
.hn-ref-title{ font-weight: 800; color: var(--hn-slate-800); margin-bottom: 6px; }

.hn-tech-grid{
  display:grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 18px;
}
@media (max-width: 1050px){
  .hn-tech-grid{ grid-template-columns: 1fr; }
}

.hn-tech{
  background: var(--hn-slate-50);
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  padding: 18px;
  transition: box-shadow 140ms ease;
}

.hn-tech:hover{ box-shadow: var(--hn-shadow-md); }
.hn-tech-title{ font-weight: 800; color: var(--hn-slate-900); margin-bottom: 8px; }

/* =====================
   Streamlit expanders (Experiments dropdowns)
   ===================== */
div[data-testid="stExpander"]{
  border: 1px solid var(--hn-border) !important;
  border-radius: var(--hn-radius-lg) !important;
  background: #ffffff !important;
  box-shadow: var(--hn-shadow-md) !important;
  overflow: hidden;
}

div[data-testid="stExpander"] details summary{
  padding: 16px 18px !important;
  font-weight: 800 !important;
  color: var(--hn-slate-900) !important;
}

div[data-testid="stExpander"] details[open] summary{
  border-bottom: 1px solid var(--hn-border) !important;
}

.hn-expander-label{
  font-weight: 800;
  color: var(--hn-slate-800);
  margin-bottom: 6px;
}

.hn-expander-body{ padding: 4px 2px 2px 2px; }
/* =====================
   Navbar
   ===================== */
.hn-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 999;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--hn-border);
  box-shadow: var(--hn-shadow-sm);
}

.hn-nav-inner{
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.hn-brand{
  display:flex;
  align-items:center;
  gap: 10px;
  font-weight: 800;
  font-size: 20px;
  color: var(--hn-slate-900);
  text-decoration: none;
}

.hn-nav-links{
  display:flex;
  align-items:center;
  gap: 28px;
}

.hn-nav-link{
  text-decoration:none;
  font-weight: 600;
  color: var(--hn-slate-600);
  transition: color 120ms ease;
}

.hn-nav-link:hover{ color: var(--hn-slate-900); }
.hn-nav-link.active, .hn-nav-link.is-active{ color: var(--hn-slate-900); font-weight: 700; }

.hn-nav-cta{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding: 10px 18px;
  border-radius: var(--hn-radius-lg);
  background: var(--hn-amber-500);
  color: var(--hn-slate-900);
  font-weight: 800;
  text-decoration:none;
  box-shadow: var(--hn-shadow-md);
  transition: background 120ms ease, box-shadow 120ms ease, transform 120ms ease;
}

.hn-nav-cta:hover{
  background: var(--hn-amber-400);
  box-shadow: var(--hn-shadow-lg);
  transform: translateY(-1px);
}

/* Provide space under fixed navbar */
.hn-nav-spacer{ height: 64px; }

/* =====================
   Tables (HTML)
   ===================== */
.hn-table-wrap{
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius-lg);
  overflow: hidden;
}

table.hn-table{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

table.hn-table thead{
  background: var(--hn-slate-50);
}

table.hn-table th{
  text-align:left;
  padding: 0.85rem 1rem;
  color: var(--hn-slate-700);
  font-weight: 700;
  border-bottom: 1px solid var(--hn-border);
  letter-spacing: 0.02em;
}

table.hn-table td{
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--hn-slate-100);
  color: var(--hn-slate-600);
}

table.hn-table tbody tr:hover{ background: var(--hn-slate-50); }

.hn-pill{
  display:inline-flex;
  align-items:center;
  gap: 0.4rem;
  font-weight: 700;
  color: var(--hn-green-500);
}

.hn-mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* Small helpers */
.hn-badge{
  display:inline-block;
  background: var(--hn-slate-100);
  color: var(--hn-slate-600);
  padding: 0.25rem 0.6rem;
  border-radius: 10px;
  font-size: 0.85rem;
}

</style>
        """,
        unsafe_allow_html=True,
    )
