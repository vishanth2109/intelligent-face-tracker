import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "database/visitors.db"

st.set_page_config(
    page_title="Intelligent Face Tracker",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #0b0f14;
        color: #f5f7fa;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0f141b;
        border-right: 1px solid #202832;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------- Header ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.25);
    }

    .brand-title {
        font-size: 27px;
        font-weight: 750;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .brand-subtitle {
        color: #8b95a5;
        font-size: 14px;
        margin-top: 3px;
    }

    /* ---------- Status ---------- */

    .status-wrapper {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        height: 100%;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #4ade80;
        padding: 8px 13px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,.8);
    }

    /* ---------- KPI Cards ---------- */

    .kpi {
        background: #111720;
        border: 1px solid #202936;
        border-radius: 16px;
        padding: 20px;
        min-height: 132px;
        box-shadow: 0 8px 30px rgba(0,0,0,.12);
    }

    .kpi-label {
        color: #8b95a5;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 31px;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .kpi-description {
        color: #697586;
        font-size: 12px;
        margin-top: 8px;
    }

    .kpi-icon {
        float: right;
        font-size: 22px;
        opacity: .9;
    }

    /* ---------- Section ---------- */

    .section-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 3px;
    }

    .section-description {
        color: #737f90;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ---------- Panels ---------- */

    .panel {
        background: #111720;
        border: 1px solid #202936;
        border-radius: 16px;
        padding: 20px;
        min-height: 100%;
    }

    /* ---------- Event badges ---------- */

    .entry-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 7px;
        background: rgba(34,197,94,.12);
        color: #4ade80;
        font-size: 11px;
        font-weight: 700;
    }

    .exit-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 7px;
        background: rgba(239,68,68,.12);
        color: #f87171;
        font-size: 11px;
        font-weight: 700;
    }

    /* ---------- Latest event ---------- */

    .event-card {
        background: #0c1118;
        border: 1px solid #202936;
        border-radius: 13px;
        padding: 15px;
        margin-top: 10px;
    }

    .event-face {
        width: 100%;
        max-height: 220px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid #27313d;
    }

    .event-face-placeholder {
        height: 190px;
        background: #151c25;
        border: 1px dashed #303b49;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #657183;
        font-size: 13px;
    }

    .event-meta {
        margin-top: 12px;
    }

    .event-face-id {
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
    }

    .event-time {
        color: #788496;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #566171;
        font-size: 11px;
        padding-top: 30px;
    }

    /* ---------- Streamlit buttons ---------- */

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #293442;
        background: #151c25;
        color: #dce2ea;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #3b82f6;
        color: #ffffff;
    }

    /* ---------- Hide default decoration ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


def get_statistics():

    conn = get_connection()

    unique_visitors = conn.execute(
        "SELECT COUNT(*) FROM persons"
    ).fetchone()[0]

    entry_events = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='ENTRY'"
    ).fetchone()[0]

    exit_events = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='EXIT'"
    ).fetchone()[0]

    active_tracks = conn.execute(
        """
        SELECT COUNT(*)
        FROM tracks
        WHERE status = 'ACTIVE'
        """
    ).fetchone()[0]

    conn.close()

    return (
        unique_visitors,
        entry_events,
        exit_events,
        active_tracks,
    )


def get_events():

    conn = get_connection()

    query = """
        SELECT
            id,
            face_id,
            track_id,
            event_type,
            timestamp,
            image_path,
            confidence
        FROM events
        ORDER BY timestamp DESC
        LIMIT 100
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def get_visitors():

    conn = get_connection()

    query = """
        SELECT
            face_id,
            first_seen,
            last_seen,
            created_at
        FROM persons
        ORDER BY created_at DESC
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([7, 2])

with header_left:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">👁</div>
            <div>
                <div class="brand-title">
                    Intelligent Face Tracker
                </div>
                <div class="brand-subtitle">
                    AI Visitor Monitoring & Recognition Platform
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:

    st.markdown(
        """
        <div class="status-wrapper">
            <div class="status">
                <span class="status-dot"></span>
                SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:20px;
            font-weight:700;
            color:white;
            margin-bottom:4px;">
            Control Center
        </div>

        <div style="
            color:#6f7b8c;
            font-size:12px;
            margin-bottom:20px;">
            Intelligent monitoring system
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🔄  Refresh Dashboard",
        use_container_width=True
    ):
        st.rerun()

    st.markdown("### System Stack")

    st.markdown(
        """
        <div style="line-height:2.1; color:#a8b1bf; font-size:13px;">
        🤖 &nbsp; YOLO Face Detection<br>
        🚶 &nbsp; ByteTrack Tracking<br>
        🧠 &nbsp; InsightFace Recognition<br>
        🗄️ &nbsp; SQLite Database<br>
        📡 &nbsp; Video / RTSP Input
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Configuration")

    st.caption("Recognition threshold")
    st.code("0.45")

    st.caption("Frame skip")
    st.code("3")

    st.caption("Tracker")
    st.code("ByteTrack")

    st.divider()

    st.caption(
        "Intelligent Face Tracker\n"
        "Hackathon Project"
    )


# ============================================================
# DATABASE CHECK
# ============================================================

if not os.path.exists(DB_PATH):

    st.error(
        f"Database not found: {DB_PATH}"
    )

    st.stop()


# ============================================================
# KPI SECTION
# ============================================================

(
    unique_visitors,
    entry_events,
    exit_events,
    active_tracks,
) = get_statistics()


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">👥</div>
            <div class="kpi-label">UNIQUE VISITORS</div>
            <div class="kpi-value">{unique_visitors}</div>
            <div class="kpi-description">
                Persistent identities
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k2:

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">🟢</div>
            <div class="kpi-label">ENTRY EVENTS</div>
            <div class="kpi-value">{entry_events}</div>
            <div class="kpi-description">
                Recorded arrivals
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k3:

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-label">EXIT EVENTS</div>
            <div class="kpi-value">{exit_events}</div>
            <div class="kpi-description">
                Recorded departures
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with k4:

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-label">ACTIVE TRACKS</div>
            <div class="kpi-value">{active_tracks}</div>
            <div class="kpi-description">
                Current tracking sessions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

events_df = get_events()
visitors_df = get_visitors()


# ============================================================
# ANALYTICS + LATEST EVENT
# ============================================================

left, right = st.columns([1.55, 1])


with left:

    st.markdown(
        '<div class="section-title">Event Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Visitor movement across the monitoring system'
        '</div>',
        unsafe_allow_html=True,
    )

    if not events_df.empty:

        chart_df = pd.DataFrame(
            {
                "Event": ["ENTRY", "EXIT"],
                "Count": [
                    int(
                        (
                            events_df["event_type"]
                            == "ENTRY"
                        ).sum()
                    ),
                    int(
                        (
                            events_df["event_type"]
                            == "EXIT"
                        ).sum()
                    ),
                ],
            }
        )

        st.bar_chart(
            chart_df.set_index("Event"),
            height=320,
        )

    else:

        st.info("No event data available.")


with right:

    st.markdown(
        '<div class="section-title">Latest Visitor Event</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Most recent recorded visitor activity'
        '</div>',
        unsafe_allow_html=True,
    )

    if not events_df.empty:

        latest = events_df.iloc[0]

        image_path = latest["image_path"]

        if (
            image_path
            and isinstance(image_path, str)
            and os.path.exists(image_path)
        ):

            st.image(
                image_path,
                use_container_width=True,
            )

        else:

            st.markdown(
                """
                <div class="event-face-placeholder">
                    Visitor image unavailable
                </div>
                """,
                unsafe_allow_html=True,
            )

        event_type = str(
            latest["event_type"]
        )

        badge = (
            '<span class="entry-badge">ENTRY</span>'
            if event_type == "ENTRY"
            else
            '<span class="exit-badge">EXIT</span>'
        )

        confidence = latest["confidence"]

        if pd.notna(confidence):

            confidence_text = (
                f"{float(confidence) * 100:.1f}%"
            )

        else:

            confidence_text = "N/A"

        st.markdown(
            f"""
            <div class="event-card">

                <div class="event-face-id">
                    {latest["face_id"]}
                </div>

                <div style="margin-top:7px;">
                    {badge}
                </div>

                <div class="event-time">
                    🕐 {latest["timestamp"]}
                </div>

                <div class="event-time">
                    🎯 Track ID: {latest["track_id"]}
                </div>

                <div class="event-time">
                    🔐 Confidence: {confidence_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info("No visitor events available.")


# ============================================================
# RECENT EVENTS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Recent Visitor Events</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Latest activity recorded by the tracking engine'
    '</div>',
    unsafe_allow_html=True,
)


if not events_df.empty:

    table_df = events_df[
        [
            "face_id",
            "track_id",
            "event_type",
            "timestamp",
            "confidence",
        ]
    ].copy()

    table_df.columns = [
        "FACE ID",
        "TRACK ID",
        "EVENT",
        "TIMESTAMP",
        "CONFIDENCE",
    ]

    table_df["CONFIDENCE"] = table_df[
        "CONFIDENCE"
    ].apply(
        lambda x:
        f"{float(x) * 100:.1f}%"
        if pd.notna(x)
        else "N/A"
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=330,
    )

else:

    st.info("No visitor events available.")


# ============================================================
# REGISTERED VISITORS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Registered Visitors</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Persistent identities stored in the recognition database'
    '</div>',
    unsafe_allow_html=True,
)


if not visitors_df.empty:

    visitor_table = visitors_df.copy()

    visitor_table.columns = [
        "FACE ID",
        "FIRST SEEN",
        "LAST SEEN",
        "REGISTERED AT",
    ]

    st.dataframe(
        visitor_table,
        use_container_width=True,
        hide_index=True,
        height=260,
    )

else:

    st.info("No registered visitors.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Intelligent Face Tracker &nbsp;•&nbsp;
        YOLO + ByteTrack + InsightFace + SQLite
        <br>
        AI-powered visitor monitoring and recognition
    </div>
    """,
    unsafe_allow_html=True,
)