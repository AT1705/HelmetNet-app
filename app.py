"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Modern Safety Theme UI

POC UPGRADE (Authority-friendly Dashboard):
- Select Malaysia location (Area) in sidebar
- Every detection logs automatically (timestamp + location + result)
- Dashboard includes:
  1) Simple KPI cards (easy wording)
  2) Malaysia hotspot map (bigger/redder = more 'no helmet' detected)
  3) Top hotspots charts + ranked table
  4) Clear recommendation panel (color-coded) for enforcement actions
  5) Recent violations list
- Optional: Create dummy logs for demo (one click)
"""

from __future__ import annotations

import tempfile
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from twilio.rest import Client
from ultralytics import YOLO
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MALAYSIA LOCATIONS (POC AREA MAP)
# ============================================================
AREA_COORDS_MY = {
    "Kuala Lumpur - Bukit Bintang": (3.1466, 101.7101),
    "Kuala Lumpur - KL Sentral": (3.1340, 101.6869),
    "Kuala Lumpur - Chow Kit": (3.1677, 101.6985),
    "Petaling Jaya": (3.1073, 101.6067),
    "Shah Alam": (3.0733, 101.5185),
    "Subang Jaya": (3.0438, 101.5800),
    "Klang": (3.0449, 101.4456),
    "Putrajaya": (2.9264, 101.6964),
    "Cyberjaya": (2.9225, 101.6506),
    "Seremban": (2.7297, 101.9381),
    "Melaka City": (2.1896, 102.2501),
    "Ipoh": (4.5975, 101.0901),
    "George Town (Penang)": (5.4141, 100.3288),
    "Alor Setar": (6.1248, 100.3676),
    "Kuantan": (3.8077, 103.3260),
    "Kota Bharu": (6.1333, 102.2386),
    "Kuala Terengganu": (5.3292, 103.1370),
    "Johor Bahru": (1.4927, 103.7414),
    "Kuching (Sarawak)": (1.5533, 110.3592),
    "Kota Kinabalu (Sabah)": (5.9804, 116.0735),
}

# ============================================================
# TWILIO TURN (Network Traversal Token -> ICE servers)
# ============================================================
@st.cache_resource
def get_twilio_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        ice_servers = token.ice_servers
        if not ice_servers:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
        return ice_servers
    except Exception as e:
        st.sidebar.error(f"TURN setup error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ICE_SERVERS = get_twilio_ice_servers()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": ICE_SERVERS})

# ============================================================
# GLOBAL CSS (Theme + Recommendation Cards)
# ============================================================
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem !important; }

    .main-header {
        font-size: 2.5rem; font-weight: 800; color: var(--text-color);
        text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center; font-size: 1.1rem; color: var(--text-color);
        opacity: 0.8; font-weight: 500; margin-bottom: 1.5rem;
    }
    h2 {
        color: var(--text-color) !important; font-weight: 700 !important;
        border-bottom: 3px solid #FFD700; padding-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary-background-color); padding: 0.5rem;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 8px; color: var(--text-color);
        font-weight: 600; padding: 0 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #FFD700 !important; color: #1E3A8A !important;
    }

    .alert-danger {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: 800;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(239,68,68,0.3);
        border: 3px solid #FCA5A5;
    }
    .alert-success {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: 800;
        margin: 20px 0; box-shadow: 0 4px 6px rgba(34,197,94,0.3);
        border: 3px solid #86EFAC;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1E3A8A; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 800;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: #1E3A8A;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; border: none;
    }

    [data-testid="metric-container"] {
        background: var(--secondary-background-color); padding: 1rem;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #FFD700;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 800 !important; color: var(--text-color);
    }

    audio {display: none;}

    .info-box {
        background: rgba(59, 130, 246, 0.10); padding: 1rem;
        border-radius: 10px; border-left: 4px solid #1E3A8A;
        margin: 1rem 0; color: var(--text-color);
    }

    /* Recommendation cards */
    .rec-wrap{
        display:grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 10px;
    }
    @media (max-width: 1100px){
        .rec-wrap{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 700px){
        .rec-wrap{ grid-template-columns: 1fr; }
    }
    .rec-card{
        border-radius: 14px;
        padding: 14px 14px;
        border: 1px solid rgba(226,232,240,1);
        box-shadow: 0 8px 18px rgba(15,23,42,0.06);
        background: white;
    }
    .rec-card.high{
        border-left: 10px solid #DC2626;
        background: linear-gradient(180deg, rgba(254,226,226,0.70), rgba(255,255,255,1));
    }
    .rec-card.med{
        border-left: 10px solid #D97706;
        background: linear-gradient(180deg, rgba(254,243,199,0.70), rgba(255,255,255,1));
    }
    .rec-card.low{
        border-left: 10px solid #16A34A;
        background: linear-gradient(180deg, rgba(220,252,231,0.70), rgba(255,255,255,1));
    }
    .rec-card.data{
        border-left: 10px solid #2563EB;
        background: linear-gradient(180deg, rgba(219,234,254,0.70), rgba(255,255,255,1));
    }
    .rec-title{
        font-weight: 900;
        font-size: 1.05rem;
        color: #0f172a;
        margin-bottom: 6px;
        line-height: 1.25;
    }
    .rec-meta{
        display:flex;
        flex-wrap:wrap;
        gap: 8px;
        margin: 8px 0 10px 0;
    }
    .rec-pill{
        font-weight: 900;
        font-size: 0.78rem;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(226,232,240,1);
        background: rgba(248,250,252,1);
        color: #334155;
    }
    .rec-body{
        color: #0f172a;
        font-weight: 650;
        font-size: 0.95rem;
        line-height: 1.35;
    }
    .rec-foot{
        margin-top: 10px;
        color: #475569;
        font-size: 0.85rem;
        font-weight: 650;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet", "nohelmet"]
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP = 3
MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = "model_1.pt"

# ============================================================
# POC COMPLIANCE LOGGING (SQLite)
# ============================================================
DB_PATH = Path("helmetnet_poc.sqlite")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        source_type TEXT NOT NULL,      -- image/video/realtime/dummy
        source_id TEXT NOT NULL,        -- filename/webrtc/seed
        area TEXT NOT NULL,             -- Malaysia location
        helmet_state TEXT NOT NULL,     -- ON/OFF/UNCERTAIN
        confidence REAL,
        duration_s REAL NOT NULL
    );
    """
    )
    conn.commit()
    conn.close()


def log_observation(
    *,
    ts: str,
    source_type: str,
    source_id: str,
    area: str,
    helmet_state: str,
    confidence: Optional[float],
    duration_s: float,
):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO observations (ts, source_type, source_id, area, helmet_state, confidence, duration_s)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, source_type, source_id, area, helmet_state, confidence, duration_s),
    )
    conn.commit()
    conn.close()


def derive_scene_state(detections: list[dict]) -> tuple[str, Optional[float]]:
    """
    POC rule:
    - OFF if any "no helmet" label exists
    - ON if there are detections and none are "no helmet"
    - UNCERTAIN if no detections
    """
    if not detections:
        return "UNCERTAIN", None
    max_conf = max(float(d.get("confidence", 0.0)) for d in detections)
    any_off = any((str(d.get("class", "")).lower() in NO_HELMET_LABELS) for d in detections)
    return ("OFF", max_conf) if any_off else ("ON", max_conf)


def load_observations_df(
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    areas: Optional[list[str]] = None,
    source_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT id, ts, source_type, source_id, area, helmet_state, confidence, duration_s FROM observations WHERE 1=1"
    params = []

    if start_ts:
        q += " AND ts >= ?"
        params.append(start_ts)
    if end_ts:
        q += " AND ts <= ?"
        params.append(end_ts)
    if areas:
        q += f" AND area IN ({','.join(['?']*len(areas))})"
        params += list(areas)
    if source_types:
        q += f" AND source_type IN ({','.join(['?']*len(source_types))})"
        params += list(source_types)

    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["bucket_hour"] = df["ts"].dt.floor("H")
    df["bucket_day"] = df["ts"].dt.floor("D")
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Layman-friendly KPIs:
    - overall_compliance: % of helmet-worn records out of helmet-worn + no-helmet
    - no_helmet_cases: how many times 'no helmet' was detected (count of OFF logs)
    - no_helmet_time_s: rough total seconds the system saw 'no helmet' (POC estimate)
    - system_not_sure_rate: % of time system couldn't decide (UNCERTAIN)
    - records_logged: how many logs exist in the selected date range/filters
    """
    if df.empty:
        return {
            "overall_compliance": None,
            "no_helmet_cases": 0,
            "no_helmet_time_s": 0.0,
            "system_not_sure_rate": None,
            "records_logged": 0,
        }

    records_logged = int(len(df))

    on_s = float(df.loc[df["helmet_state"] == "ON", "duration_s"].sum())
    off_s = float(df.loc[df["helmet_state"] == "OFF", "duration_s"].sum())
    un_s = float(df.loc[df["helmet_state"] == "UNCERTAIN", "duration_s"].sum())
    total_s = on_s + off_s + un_s

    known = on_s + off_s
    overall_compliance = (on_s / known) if known > 0 else None
    system_not_sure_rate = (un_s / total_s) if total_s > 0 else None

    no_helmet_cases = int((df["helmet_state"] == "OFF").sum())

    return {
        "overall_compliance": overall_compliance,
        "no_helmet_cases": no_helmet_cases,
        "no_helmet_time_s": off_s,
        "system_not_sure_rate": system_not_sure_rate,
        "records_logged": records_logged,
    }


def aggregate_by_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a per-area summary that supports:
    - map sizing (no_helmet_time_s)
    - top hotspot charts (no_helmet_cases + no_helmet_rate)
    - recommendation rules
    """
    if df.empty:
        return df

    # seconds (duration) by state
    sec = df.groupby(["area", "helmet_state"])["duration_s"].sum().unstack(fill_value=0)

    # counts (events) by state
    cnt = df.groupby(["area", "helmet_state"]).size().unstack(fill_value=0)

    for col in ["ON", "OFF", "UNCERTAIN"]:
        if col not in sec.columns:
            sec[col] = 0.0
        if col not in cnt.columns:
            cnt[col] = 0

    out = pd.DataFrame({"area": sec.index})
    out["helmet_time_s"] = sec["ON"].astype(float)
    out["no_helmet_time_s"] = sec["OFF"].astype(float)            # rough
    out["not_sure_time_s"] = sec["UNCERTAIN"].astype(float)

    out["helmet_cases"] = cnt["ON"].astype(int)
    out["no_helmet_cases"] = cnt["OFF"].astype(int)
    out["not_sure_cases"] = cnt["UNCERTAIN"].astype(int)

    known_cases = out["helmet_cases"] + out["no_helmet_cases"]
    out["overall_compliance"] = out["helmet_cases"] / known_cases.replace({0: pd.NA})
    out["no_helmet_rate"] = out["no_helmet_cases"] / known_cases.replace({0: pd.NA})

    total_cases = out["helmet_cases"] + out["no_helmet_cases"] + out["not_sure_cases"]
    out["system_not_sure_rate"] = out["not_sure_cases"] / total_cases.replace({0: pd.NA})

    return out.reset_index(drop=True)


def trend_over_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    bucket_col = "bucket_hour" if freq == "H" else "bucket_day"
    g = df.groupby([bucket_col, "helmet_state"])["duration_s"].sum().unstack(fill_value=0).reset_index()
    for col in ["ON", "OFF", "UNCERTAIN"]:
        if col not in g.columns:
            g[col] = 0.0
    g["known_s"] = g["ON"] + g["OFF"]
    g["overall_compliance"] = g["ON"] / g["known_s"].replace({0: pd.NA})
    g.rename(columns={bucket_col: "bucket"}, inplace=True)
    return g


def render_recommendations(by_area: pd.DataFrame) -> None:
    """
    Color-coded recommendation panel for authorities.
    We prioritize:
    - Data Quality Fix if system_not_sure_rate is high
    - Otherwise enforce at places with high no_helmet_rate + significant volume
    """
    if by_area is None or by_area.empty:
        st.info("No recommendations available (no data).")
        return

    # pick top 6 by "no_helmet_cases" primarily, then by "no_helmet_rate"
    cand = by_area.copy()
    cand["no_helmet_rate_num"] = cand["no_helmet_rate"].astype("float")
    cand["overall_compliance_num"] = cand["overall_compliance"].astype("float")
    cand["system_not_sure_rate_num"] = cand["system_not_sure_rate"].astype("float")
    cand = cand.sort_values(["no_helmet_cases", "no_helmet_rate_num"], ascending=[False, False]).head(6)

    cards = []
    for _, r in cand.iterrows():
        loc = str(r["area"])
        nh_cases = int(r["no_helmet_cases"])
        nh_rate = r["no_helmet_rate_num"]
        comp = r["overall_compliance_num"]
        unsure = r["system_not_sure_rate_num"]
        nh_time = float(r["no_helmet_time_s"])

        nh_rate_txt = "-" if pd.isna(nh_rate) else f"{nh_rate*100:.1f}%"
        comp_txt = "-" if pd.isna(comp) else f"{comp*100:.1f}%"
        unsure_txt = "-" if pd.isna(unsure) else f"{unsure*100:.1f}%"

        # Rules (POC):
        # 1) If unsure > 30% -> data quality
        # 2) Else if no-helmet rate >= 40% and cases >= 5 -> high enforcement
        # 3) Else if no-helmet rate >= 20% and cases >= 3 -> medium
        # 4) Else -> low
        card_class = "low"
        headline = "Low Priority"
        action = "Monitor and review weekly."

        if (not pd.isna(unsure)) and float(unsure) >= 0.30:
            card_class = "data"
            headline = "Improve Camera / Data Quality"
            action = "Improve camera angle/lighting; reduce blur; verify model confidence before enforcement."
        else:
            if (not pd.isna(nh_rate)) and float(nh_rate) >= 0.40 and nh_cases >= 5:
                card_class = "high"
                headline = "High Priority Enforcement"
                action = "Deploy patrol or enforcement checkpoint during peak hours; conduct targeted safety operations."
            elif (not pd.isna(nh_rate)) and float(nh_rate) >= 0.20 and nh_cases >= 3:
                card_class = "med"
                headline = "Targeted Enforcement"
                action = "Increase periodic patrols; add warning signage and public awareness; schedule follow-up checks."

        cards.append(
            f"""
            <div class="rec-card {card_class}">
              <div class="rec-title">{loc}<br/><span style="font-weight:800; color:#334155;">{headline}</span></div>
              <div class="rec-meta">
                <span class="rec-pill">No-Helmet Cases: {nh_cases}</span>
                <span class="rec-pill">No-Helmet Rate: {nh_rate_txt}</span>
                <span class="rec-pill">Compliance: {comp_txt}</span>
                <span class="rec-pill">System Not Sure: {unsure_txt}</span>
              </div>
              <div class="rec-body">{action}</div>
              <div class="rec-foot">POC metric note: “No-Helmet Time” is a rough estimate from sampled frames (video/live).</div>
            </div>
            """
        )

    html = f"""
    <div class="rec-wrap">
      {''.join(cards)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


init_db()

# ============================================================
# UTILS & LOGIC
# ============================================================
@st.cache_resource
def load_model(model_file: str):
    try:
        candidate = MODELS_DIR / model_file
        if candidate.exists():
            model = YOLO(str(candidate))
            st.sidebar.success(f"✅ Model loaded: {model_file}")
            return model
        st.sidebar.warning("⚠️ Model not found in ./models, using YOLOv8n")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


def play_alarm():
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0.0
    if time.time() - st.session_state.last_alarm > 3:
        if Path("alert.mp3").exists():
            st.audio("alert.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()


def draw_boxes(frame, detections):
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = (0, 0, 139) if det["class"] in NO_HELMET_LABELS else (0, 100, 0)
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def detect_frame(frame, model, conf_threshold):
    results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False, device="cpu")

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = str(model.names[cls_id]).lower()
        conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()

        detections.append({"class": cls_name, "confidence": conf, "bbox": bbox})

        if cls_name in NO_HELMET_LABELS:
            no_helmet_count += 1
        else:
            helmet_count += 1

    return detections, {
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "alert": no_helmet_count > 0,
    }


def render_detection_table(detections: list[dict], model_name: str) -> None:
    """
    Pretty results card + table. Uses components.html, so we inline CSS (components are isolated).
    """
    css = """
    <style>
      :root { --border:#e2e8f0; --muted:#64748b; --text:#0f172a; --bg:#ffffff; }
      body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--text); background:transparent; }
      .card{ background:var(--bg); border:1px solid var(--border); border-radius:16px; box-shadow:0 12px 30px rgba(15,23,42,.08); overflow:hidden; }
      .head{ padding:16px 18px; border-bottom:1px solid var(--border); display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
      .title{ font-weight:900; font-size:18px; line-height:1.15; }
      .sub{ margin-top:4px; color:var(--muted); font-size:13px; font-weight:600; }
      .pill{ margin-top:2px; padding:8px 12px; border-radius:999px; border:1px solid var(--border); background:#f8fafc; font-weight:900; font-size:13px; color:#334155; white-space:nowrap; }
      .section{ padding:14px 18px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
      .section .left{ font-weight:900; font-size:16px; }
      .section .right{ color:var(--muted); font-weight:800; font-size:13px; }
      .wrap{ overflow:auto; }
      table{ width:100%; border-collapse:collapse; min-width:820px; background:#fff; }
      thead th{ text-align:left; padding:12px 14px; font-size:12px; letter-spacing:.02em; color:#475569; font-weight:900;
                background:#f8fafc; border-top:1px solid #eef2f7; border-bottom:1px solid #eef2f7; }
      tbody td{ padding:12px 14px; border-bottom:1px solid #eef2f7; font-size:14px; vertical-align:middle; }
      tbody tr:nth-child(odd){ background:#ffffff; }
      tbody tr:nth-child(even){ background:#fbfdff; }
      .num{ color:#475569; width:46px; }
      .label{ font-weight:900; }
      .bbox{ color:#475569; font-variant-numeric: tabular-nums; }
      .ok{ display:inline-flex; align-items:center; justify-content:center; padding:6px 12px; border-radius:999px;
           background:#dcfce7; color:#14532d; font-weight:900; font-size:13px; border:1px solid #86efac; }
      .bad{ display:inline-flex; align-items:center; justify-content:center; padding:6px 12px; border-radius:999px;
            background:#fee2e2; color:#991b1b; font-weight:900; font-size:13px; border:1px solid #fca5a5; }
      .foot{ padding:12px 18px; border-top:1px solid var(--border); color:var(--muted); font-size:13px; font-weight:600; background:#ffffff; }
    </style>
    """

    def badge_html(label: str) -> str:
        lab = (label or "").lower()
        non = lab in NO_HELMET_LABELS or lab.replace("_", "-") in NO_HELMET_LABELS
        return '<span class="bad">No Helmet</span>' if non else '<span class="ok">Helmet</span>'

    if not detections:
        html = f"""
        {css}
        <div class="card">
          <div class="head">
            <div>
              <div class="title">Results</div>
              <div class="sub">No detections found.</div>
            </div>
            <div class="pill">Model: {model_name}</div>
          </div>
          <div class="foot">Tip: Try a clearer image for better detection.</div>
        </div>
        """
        components.html(html, height=200, scrolling=False)
        return

    dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

    rows = []
    for i, det in enumerate(dets, start=1):
        label = str(det.get("class", ""))
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", [0, 0, 0, 0])

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
        except Exception:
            x1 = y1 = w = h = 0

        rows.append(
            f"""
            <tr>
              <td class="num">{i}</td>
              <td class="label">{label}</td>
              <td>{conf*100:.1f}%</td>
              <td>{badge_html(label)}</td>
              <td class="bbox">{x1}, {y1}, {w}, {h}</td>
            </tr>
            """
        )

    height = 520 if len(rows) <= 6 else 620

    html = f"""
    {css}
    <div class="card">
      <div class="head">
        <div>
          <div class="title">Results</div>
          <div class="sub">YOLO output (label, confidence, bbox).</div>
        </div>
        <div class="pill">Model: {model_name}</div>
      </div>

      <div class="section">
        <div class="left">Detections Table</div>
        <div class="right">Sorted by confidence</div>
      </div>

      <div class="wrap">
        <table>
          <thead>
            <tr>
              <th style="width:56px;">#</th>
              <th>LABEL</th>
              <th>CONFIDENCE</th>
              <th>STATUS</th>
              <th>BBOX (X,Y,W,H)</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>

      <div class="foot">
        Tip: Each run is automatically saved to Dashboard with the selected Malaysia location.
      </div>
    </div>
    """
    components.html(html, height=height, scrolling=True)

# ============================================================
# WEBRTC CLASS
# ============================================================
class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.helmet = 0
        self.no_helmet = 0
        self.frame_cnt = 0
        self.last_dets = []
        self.alert = False

        # logging throttle
        self.last_log_ts = 0.0
        self.area = "Unknown"
        self.source_id = "webrtc"

    def set_model(self, model, conf, area="Unknown", source_id="webrtc"):
        self.model = model
        self.conf = conf
        self.area = area
        self.source_id = source_id

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return img

        self.frame_cnt += 1
        if self.frame_cnt % FRAME_SKIP == 0:
            try:
                detections, stats = detect_frame(img, self.model, self.conf)
                self.last_dets = detections
                self.helmet = stats["helmet_count"]
                self.no_helmet = stats["no_helmet_count"]
                self.alert = stats["alert"]

                now = time.time()
                if now - self.last_log_ts >= 1.0:
                    helmet_state, state_conf = derive_scene_state(detections)
                    log_observation(
                        ts=utc_now_iso(),
                        source_type="realtime",
                        source_id=self.source_id,
                        area=self.area,
                        helmet_state=helmet_state,
                        confidence=state_conf,
                        duration_s=1.0,
                    )
                    self.last_log_ts = now
            except Exception:
                pass

        return draw_boxes(img, self.last_dets)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🤖 Model Settings**")
    model_files = sorted([p.name for p in MODELS_DIR.glob("*.pt")])
    if not model_files:
        st.warning("No .pt files found in ./models. Falling back to yolov8n.pt")
        model_files = ["yolov8n.pt"]

    default_index = model_files.index(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH in model_files else 0
    model_choice = st.selectbox("Select Model", options=model_files, index=default_index)
    confidence_threshold = st.slider("🎯 Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("**📍 Location (Malaysia)**")
    if "area_list" not in st.session_state:
        st.session_state.area_list = list(AREA_COORDS_MY.keys())
    current_area = st.selectbox("Select Location", st.session_state.area_list, key="current_area")

    st.markdown("---")
    st.markdown("**📊 Demo Tools (POC)**")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Create Dummy Data", key="dummy_logs"):
            now = datetime.now(timezone.utc)
            areas = st.session_state.area_list[:]
            for i in range(180):
                ts = (now - timedelta(minutes=180 - i)).isoformat()
                area = areas[i % len(areas)]
                # Bias for demo hotspots:
                if "Bukit Bintang" in area or "Chow Kit" in area:
                    state = "OFF" if i % 2 == 0 else "ON"
                elif "Johor Bahru" in area or "George Town" in area:
                    state = "OFF" if i % 4 == 0 else "ON"
                else:
                    state = "OFF" if i % 10 == 0 else "ON"

                log_observation(
                    ts=ts,
                    source_type="dummy",
                    source_id="seed",
                    area=area,
                    helmet_state=state,
                    confidence=0.85,
                    duration_s=1.0,
                )
            st.success("Dummy data created. Open the Dashboard tab.")

    with c2:
        if st.button("Clear Logs", key="clear_logs"):
            conn = get_conn()
            conn.execute("DELETE FROM observations")
            conn.commit()
            conn.close()
            st.warning("All logs cleared.")

    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total Detections (this session)", st.session_state.total_detections)
    st.caption(f"DB File: {DB_PATH.name}")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def _load_yolo(choice: str):
    if choice == "yolov8n.pt":
        return YOLO("yolov8n.pt")
    m = load_model(choice)
    return m if m else YOLO("yolov8n.pt")

model = _load_yolo(model_choice)

# ============================================================
# MAIN APP UI
# ============================================================
st.markdown('<h1 class="main-header">HelmetNet</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI Helmet Detection System</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Image Detection", "Video Detection", "Real-Time Detection", "Dashboard (Hotspots)"]
)

# --- TAB 1: IMAGE DETECTION ---
with tab1:
    st.markdown("### Upload an Image")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<div class="info-box"><strong>Tips:</strong><br>• Clear, well-lit images<br>• JPG, PNG, BMP</div>',
            unsafe_allow_html=True,
        )

    with col1:
        img_file = st.file_uploader(
            "Choose image",
            ["jpg", "jpeg", "png", "bmp"],
            key="img",
            label_visibility="collapsed",
        )

    run_img = st.button("Run Detection", key="run_img")

    if img_file and run_img:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Analyzing..."):
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)

            helmet_state, state_conf = derive_scene_state(dets)
            log_observation(
                ts=utc_now_iso(),
                source_type="image",
                source_id=str(img_file.name),
                area=st.session_state.get("current_area", "Unknown"),
                helmet_state=helmet_state,
                confidence=state_conf,
                duration_s=1.0,
            )

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Original**")
            st.image(original_rgb, use_container_width=True)
        with c2:
            st.markdown("**Result**")
            st.image(annotated_rgb, use_container_width=True)

        st.info(f"Saved to Dashboard as location: {st.session_state.get('current_area','Unknown')}")

        if stats["alert"]:
            st.markdown('<div class="alert-danger">NO HELMET DETECTED</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">NO VIOLATION DETECTED</div>', unsafe_allow_html=True)

        st.markdown("### Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Helmets", stats["helmet_count"])
        m2.metric("No Helmets", stats["no_helmet_count"])
        m3.metric("Total Objects", len(dets))

        render_detection_table(dets, model_choice)

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, "rb") as f:
            st.download_button("Download Result", f, f"result_{img_file.name}", "image/jpeg")

# --- TAB 2: VIDEO DETECTION ---
with tab2:
    st.markdown("### Upload a Video")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<div class="info-box"><strong>Fast Mode:</strong><br>• Frame skipping<br>• Live preview<br>• MP4, AVI, MOV</div>',
            unsafe_allow_html=True,
        )

    with col1:
        vid_file = st.file_uploader(
            "Choose video",
            ["mp4", "avi", "mov", "mkv"],
            key="vid",
            label_visibility="collapsed",
        )

    if vid_file:
        st.markdown("### Processing")
        if st.button("Start Live Inference", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))

            st_frame = st.empty()
            st_metrics = st.empty()
            st_progress = st.progress(0)

            frame_count = 0
            cached_detections = []
            current_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

            sample_duration_s = float(FRAME_SKIP / max(fps, 1.0))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                ran_inference = False

                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)
                    ran_inference = True

                annotated = draw_boxes(frame, cached_detections)
                out.write(annotated)

                if current_stats["alert"]:
                    play_alarm()

                if ran_inference:
                    helmet_state, state_conf = derive_scene_state(cached_detections)
                    log_observation(
                        ts=utc_now_iso(),
                        source_type="video",
                        source_id=str(vid_file.name),
                        area=st.session_state.get("current_area", "Unknown"),
                        helmet_state=helmet_state,
                        confidence=state_conf,
                        duration_s=sample_duration_s,
                    )

                st_frame.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_count}/{total_frames}" if total_frames else f"Frame {frame_count}",
                    use_container_width=True,
                )

                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Helmets", current_stats["helmet_count"])
                    c2.metric("Violations", current_stats["no_helmet_count"])
                    c3.metric("Progress", f"{int(frame_count/total_frames*100)}%" if total_frames else "-")

                if total_frames:
                    st_progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()

            st.success("Processing Complete!")
            st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

            with open(outfile.name, "rb") as f:
                st.download_button("Download Result Video", f, "result.mp4", "video/mp4")

# --- TAB 3: REAL-TIME DETECTION (WEBRTC) ---
with tab3:
    st.markdown("### Real-Time Live Detection")
    st.markdown(
        """
    <div class="info-box">
    <strong>Live Webcam:</strong><br>
    • Click "START"<br>
    • Logs to dashboard about once per second<br>
    </div>
    """,
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HelmetTransformer,
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(
            model,
            confidence_threshold,
            area=st.session_state.get("current_area", "Unknown"),
            source_id="webrtc",
        )

        st.markdown("### Live Stats")
        m1, m2 = st.columns(2)
        m1.metric("Helmets", ctx.video_processor.helmet)
        m2.metric("Violations", ctx.video_processor.no_helmet)

        if ctx.video_processor.alert:
            st.markdown('<div class="alert-danger">NO HELMET DETECTED</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">NO VIOLATION DETECTED</div>', unsafe_allow_html=True)

# --- TAB 4: DASHBOARD (HOTSPOTS) ---
with tab4:
    st.markdown("### Malaysia Helmet Violation Hotspots (POC)")

    st.markdown(
        '<div class="info-box"><strong>How to read:</strong><br>'
        '• Bigger / redder points = more no-helmet detected at that location<br>'
        '• Use the recommendations section to decide enforcement actions</div>',
        unsafe_allow_html=True,
    )

    # default: last 24 hours
    now = datetime.now(timezone.utc)
    default_start = (now - timedelta(hours=24)).date()
    default_end = now.date()

    colf1, colf2, colf3 = st.columns([1, 1, 2])
    with colf1:
        start_date = st.date_input("Start date", value=default_start, key="dash_start")
    with colf2:
        end_date = st.date_input("End date", value=default_end, key="dash_end")
    with colf3:
        areas = st.multiselect(
            "Locations",
            st.session_state.area_list,
            default=st.session_state.area_list,
            key="dash_areas",
        )

    source_types = st.multiselect(
        "Sources",
        ["image", "video", "realtime", "dummy"],
        default=["image", "video", "realtime", "dummy"],
        key="dash_src",
    )
    freq = st.selectbox("Trend view", ["H", "D"], index=0, key="dash_freq")

    start_ts = pd.Timestamp(start_date).tz_localize("UTC").isoformat()
    end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC").isoformat()

    df = load_observations_df(start_ts=start_ts, end_ts=end_ts, areas=areas, source_types=source_types)
    k = compute_kpis(df)
    by_area = aggregate_by_area(df) if not df.empty else pd.DataFrame()

    # KPI strip (easy wording)
    worst_location = "-"
    worst_cases = 0
    if not by_area.empty:
        w = by_area.sort_values(["no_helmet_cases", "no_helmet_rate"], ascending=[False, False]).iloc[0]
        worst_location = str(w["area"])
        worst_cases = int(w["no_helmet_cases"])

    # compare to previous equal-length period
    period_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)
    prev_end = pd.Timestamp(start_date).tz_localize("UTC") - pd.Timedelta(seconds=1)
    prev_start = prev_end - pd.Timedelta(days=period_days) + pd.Timedelta(seconds=1)
    df_prev = load_observations_df(
        start_ts=prev_start.isoformat(),
        end_ts=prev_end.isoformat(),
        areas=areas,
        source_types=source_types,
    )
    k_prev = compute_kpis(df_prev)
    change_cases = k["no_helmet_cases"] - k_prev["no_helmet_cases"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total No-Helmet Cases", f"{k['no_helmet_cases']}")
    m2.metric("Worst Location", worst_location)
    m3.metric("Overall Helmet Compliance", "-" if k["overall_compliance"] is None else f"{k['overall_compliance']*100:.1f}%")
    m4.metric("Change vs Previous Period", f"{change_cases:+d} cases")

    with st.expander("Meaning of the KPIs (simple terms)"):
        st.markdown(
            """
- **Total No-Helmet Cases**: how many times the system detected someone without a helmet.
- **Worst Location**: the location with the highest number of no-helmet detections.
- **Overall Helmet Compliance**: percentage of “helmet detected” compared to “helmet + no-helmet”.
- **Change vs Previous Period**: difference in no-helmet cases compared to the previous time window.
- **System Not Sure**: how often the system cannot decide (poor lighting/blur/far objects). High value suggests camera improvements.
            """
        )

    # ---------------- MAP ----------------
    st.markdown("#### Hotspot Map")
    if df.empty or by_area.empty:
        st.info("No data yet. Run detection or click 'Create Dummy Data' in sidebar.")
    else:
        map_df = by_area.copy()
        map_df["lat"] = map_df["area"].apply(lambda a: AREA_COORDS_MY.get(a, (None, None))[0])
        map_df["lon"] = map_df["area"].apply(lambda a: AREA_COORDS_MY.get(a, (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"])

        # risk weighting by no-helmet cases (simple + easy to explain)
        max_cases = float(map_df["no_helmet_cases"].max()) if float(map_df["no_helmet_cases"].max()) > 0 else 1.0
        map_df["risk_score"] = map_df["no_helmet_cases"] / max_cases
        map_df["radius"] = map_df["risk_score"] * 8000 + 1500

        # color ramp: high risk -> more red
        map_df["color"] = map_df["risk_score"].apply(lambda r: [255, int(255 * (1 - r)), int(255 * (1 - r))])

        import pydeck as pdk

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )

        def pct(x):
            return "-" if pd.isna(x) else f"{float(x)*100:.1f}%"

        map_df["compliance_text"] = map_df["overall_compliance"].apply(pct)
        map_df["no_helmet_rate_text"] = map_df["no_helmet_rate"].apply(pct)

        tooltip = {
            "html": "<b>{area}</b><br/>No-Helmet Cases: {no_helmet_cases}<br/>No-Helmet Rate: {no_helmet_rate_text}<br/>Compliance: {compliance_text}",
            "style": {"backgroundColor": "white", "color": "black"},
        }

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=4.2105, longitude=101.9758, zoom=5, pitch=0),
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, use_container_width=True)

    # ---------------- TOP HOTSPOTS VISUALS ----------------
    st.markdown("#### Top Hotspots (Visual)")

    if not by_area.empty:
        topN = by_area.sort_values(["no_helmet_cases", "no_helmet_rate"], ascending=[False, False]).head(10).copy()
        topN = topN.set_index("area")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**No-Helmet Cases by Location (Top 10)**")
            st.bar_chart(topN["no_helmet_cases"])

        with cB:
            st.markdown("**No-Helmet Rate by Location (Top 10)**")
            # show % as 0-1 numeric (streamlit will chart it)
            st.bar_chart(topN["no_helmet_rate"])

        st.markdown("#### Top Hotspots (Ranked Table)")
        table = topN.reset_index()[[
            "area",
            "no_helmet_cases",
            "no_helmet_rate",
            "overall_compliance",
            "system_not_sure_rate",
        ]].copy()

        table["no_helmet_rate"] = table["no_helmet_rate"].apply(lambda x: "-" if pd.isna(x) else f"{float(x)*100:.1f}%")
        table["overall_compliance"] = table["overall_compliance"].apply(lambda x: "-" if pd.isna(x) else f"{float(x)*100:.1f}%")
        table["system_not_sure_rate"] = table["system_not_sure_rate"].apply(lambda x: "-" if pd.isna(x) else f"{float(x)*100:.1f}%")

        table = table.rename(columns={
            "area": "Location",
            "no_helmet_cases": "No-Helmet Cases",
            "no_helmet_rate": "No-Helmet Rate",
            "overall_compliance": "Helmet Compliance",
            "system_not_sure_rate": "System Not Sure",
        })

        st.dataframe(table, use_container_width=True)

    # ---------------- RECOMMENDATIONS (COLOR-CODED) ----------------
    st.markdown("#### Recommended Actions (Easy to Read)")
    render_recommendations(by_area)

    # ---------------- TREND ----------------
    st.markdown("#### Trend (Helmet Compliance Over Time)")
    tr = trend_over_time(df, freq=freq)
    if tr.empty:
        st.info("No trend data available for the selected range.")
    else:
        tr2 = tr.set_index("bucket")[["overall_compliance", "OFF", "UNCERTAIN"]].copy()
        st.line_chart(tr2[["overall_compliance"]])
        st.line_chart(tr2[["OFF", "UNCERTAIN"]])

    # ---------------- RECENT VIOLATIONS ----------------
    st.markdown("#### Recent No-Helmet Cases (Latest 20)")
    if df.empty:
        st.write("No data.")
    else:
        recent = df[df["helmet_state"] == "OFF"].sort_values("ts", ascending=False).head(20).copy()
        if recent.empty:
            st.write("No no-helmet cases in the selected range.")
        else:
            recent["Time (UTC)"] = recent["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
            recent_table = recent[["Time (UTC)", "area", "source_type", "source_id", "confidence"]].rename(
                columns={"area": "Location", "source_type": "Source", "source_id": "File/Camera", "confidence": "Confidence"}
            )
            st.dataframe(recent_table, use_container_width=True)

st.markdown("---")
st.caption("HelmetNet App | Malaysia Hotspot Dashboard with Recommendations (POC)")
