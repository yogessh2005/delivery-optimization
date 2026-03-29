import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import io
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# ─────────────────────────────────────────────────────────
# CONSTANTS

WAREHOUSE_LAT = 12.9716   # Bangalore warehouse (default)
WAREHOUSE_LON = 77.5946

PRIORITY_ORDER  = {"High": 0, "Medium": 1, "Low": 2}
PRIORITY_WEIGHT = {"High": 1.0, "Medium": 1.5, "Low": 2.0}  # multiplier for weighted dist
AGENTS          = ["Agent 1", "Agent 2", "Agent 3"]

AGENT_COLORS = ["#E63946", "#2A9D8F", "#E9C46A"]

# ─────────────────────────────────────────────────────────
# MODULE 1 — read_csv

def read_csv(file, force_small=False) -> pd.DataFrame:
    """
    Read and validate a delivery CSV file.

    Accepts a file-like object (from Streamlit uploader) or a path string.
    Handles missing columns, bad values, and computes Distance if coordinates exist.

    Returns a clean DataFrame with columns:
        Location ID, Distance from warehouse, Delivery Priority
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Cannot parse CSV: {e}")

    # ── Normalise column names (strip spaces, title-case) ──
    df.columns = df.columns.str.strip()

    # ── Ensure Location ID exists ──
    loc_candidates = [c for c in df.columns if "location" in c.lower() or "id" in c.lower()]
    if "Location ID" not in df.columns:
        if loc_candidates:
            df.rename(columns={loc_candidates[0]: "Location ID"}, inplace=True)
        else:
            df.insert(0, "Location ID", [f"LOC{str(i+1).zfill(3)}" for i in range(len(df))])

    df["Location ID"] = df["Location ID"].astype(str).str.strip()
    df.drop_duplicates(subset="Location ID", inplace=True)

    if force_small:
        for c in list(df.columns):
            if "lat" in c.lower() or "lon" in c.lower() or "lng" in c.lower() or "distance from warehouse" in c.lower():
                df.drop(columns=[c], inplace=True)

    # ── Compute distance from coordinates if available ──
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)

    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df["Distance from warehouse"] = df.apply(
            lambda r: haversine(WAREHOUSE_LAT, WAREHOUSE_LON, r[lat_col], r[lon_col])
            if pd.notna(r[lat_col]) and pd.notna(r[lon_col]) else np.nan,
            axis=1
        )
    elif "Distance from warehouse" not in df.columns:
        # Generate synthetic small distances (2–35 km)
        rng = np.random.default_rng(42)
        df["Distance from warehouse"] = rng.uniform(2, 35, size=len(df)).round(2)

    # ── Fill missing distances with column median ──
    df["Distance from warehouse"] = pd.to_numeric(
        df["Distance from warehouse"], errors="coerce"
    )
    med = df["Distance from warehouse"].median()
    df["Distance from warehouse"].fillna(med if pd.notna(med) else 20.0, inplace=True)
    df["Distance from warehouse"] = df["Distance from warehouse"].round(2)

    # ── Ensure Delivery Priority exists ──
    if "Delivery Priority" not in df.columns:
        rng2 = np.random.default_rng(7)
        df["Delivery Priority"] = rng2.choice(
            ["High", "Medium", "Low"], size=len(df), p=[0.3, 0.4, 0.3]
        )
    else:
        valid = {"High", "Medium", "Low"}
        mask = ~df["Delivery Priority"].isin(valid)
        if mask.any():
            rng3 = np.random.default_rng(99)
            df.loc[mask, "Delivery Priority"] = rng3.choice(["High", "Medium", "Low"], size=mask.sum())

    # ── Drop rows still unusable ──
    df.dropna(subset=["Location ID", "Distance from warehouse"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ─────────────────────────────────────────────────────────
# MODULE 2 — sort_deliveries
# ─────────────────────────────────────────────────────────

def sort_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort deliveries by priority (High > Medium > Low) then by ascending distance.

    Returns a sorted copy of the DataFrame.
    """
    df = df.copy()
    df["_priority_rank"] = df["Delivery Priority"].map(PRIORITY_ORDER)
    df.sort_values(["_priority_rank", "Distance from warehouse"], inplace=True)
    df.drop(columns=["_priority_rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────
# MODULE 3 — assign_deliveries
# ─────────────────────────────────────────────────────────

def assign_deliveries(df: pd.DataFrame, use_weighted: bool = False) -> dict:
    """
    Greedy load-balancing assignment of deliveries to 3 agents.

    Algorithm:
        For each delivery (sorted by priority then distance):
            Assign to the agent whose current total distance (or weighted distance) is lowest.

    Parameters:
        df           : sorted deliveries DataFrame
        use_weighted : if True, weight distances by priority to favour High-priority routes

    Returns a dict:
        {
            "Agent 1": {"locations": [...], "distances": [...], "priorities": [...], "total": float},
            ...
        }
    """
    agents = {name: {"locations": [], "distances": [], "priorities": [], "total": 0.0}
              for name in AGENTS}

    for _, row in df.iterrows():
        loc      = row["Location ID"]
        dist     = float(row["Distance from warehouse"])
        priority = row["Delivery Priority"]
        weight   = PRIORITY_WEIGHT[priority] if use_weighted else 1.0

        # Pick agent with lowest effective total
        best = min(agents, key=lambda a: agents[a]["total"])
        agents[best]["locations"].append(loc)
        agents[best]["distances"].append(dist)
        agents[best]["priorities"].append(priority)
        agents[best]["total"] = round(agents[best]["total"] + dist * weight, 2)

    return agents


# ─────────────────────────────────────────────────────────
# MODULE 4 — export_plan
# ─────────────────────────────────────────────────────────

def export_plan(assignment: dict) -> pd.DataFrame:
    """
    Convert agent assignment dict to a clean exportable DataFrame.

    Returns a DataFrame with columns:
        Agent | Location IDs | Total Distance (km) | Delivery Count
    """
    rows = []
    for agent, data in assignment.items():
        rows.append({
            "Agent":               agent,
            "Location IDs":        ", ".join(data["locations"]),
            "Total Distance (km)": data["total"],
            "Delivery Count":      len(data["locations"]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# MODULE 5 — visualize_results
# ─────────────────────────────────────────────────────────

def visualize_results(assignment: dict) -> plt.Figure:
    """
    Generate a styled bar chart comparing total distance per agent.

    Returns a matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0F1117")

    agents    = list(assignment.keys())
    totals    = [assignment[a]["total"] for a in agents]
    counts    = [len(assignment[a]["locations"]) for a in agents]

    # ── Chart 1: Total Distance ──
    ax1 = axes[0]
    ax1.set_facecolor("#1A1D27")
    bars = ax1.bar(agents, totals, color=AGENT_COLORS, width=0.5, zorder=3)
    ax1.set_title("Total Distance per Agent (km)", color="white", fontsize=13, pad=12)
    ax1.set_ylabel("Distance (km)", color="#AAAAAA", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333344")
    ax1.yaxis.set_tick_params(labelcolor="#AAAAAA")
    ax1.xaxis.set_tick_params(labelcolor="white")
    ax1.set_ylim(0, max(totals) * 1.25)
    ax1.yaxis.grid(True, color="#333344", linestyle="--", alpha=0.5, zorder=0)
    for bar, val in zip(bars, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.02,
                 f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    # ── Chart 2: Delivery Count ──
    ax2 = axes[1]
    ax2.set_facecolor("#1A1D27")
    bars2 = ax2.bar(agents, counts, color=AGENT_COLORS, width=0.5, zorder=3)
    ax2.set_title("Deliveries Assigned per Agent", color="white", fontsize=13, pad=12)
    ax2.set_ylabel("# Deliveries", color="#AAAAAA", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333344")
    ax2.yaxis.set_tick_params(labelcolor="#AAAAAA")
    ax2.xaxis.set_tick_params(labelcolor="white")
    ax2.set_ylim(0, max(counts) * 1.25)
    ax2.yaxis.grid(True, color="#333344", linestyle="--", alpha=0.5, zorder=0)
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                 str(val), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=2.5)
    return fig


# ─────────────────────────────────────────────────────────
# HELPER — Haversine distance
# ─────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Return distance in km between two (lat, lon) points."""
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ     = radians(lat2 - lat1)
    dλ     = radians(lon2 - lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return round(2*R*atan2(sqrt(a), sqrt(1-a)), 2)


# ─────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────

def main():
    global WAREHOUSE_LAT, WAREHOUSE_LON
    
    # ── Page config ──
    st.set_page_config(
        page_title="Delivery Optimizer",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F1117;
        color: #E8E8F0;
    }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(120deg, #E63946 0%, #E9C46A 50%, #2A9D8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #888899;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2A2D3E;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .metric-label { color: #888899; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; }
    .agent-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .agent-1 { background: #1E1013; border-color: #E63946; }
    .agent-2 { background: #0E1A18; border-color: #2A9D8F; }
    .agent-3 { background: #1E1B0A; border-color: #E9C46A; }
    .badge-high   { background: #E63946; color: white; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 600; }
    .badge-medium { background: #E9C46A; color: #1A1200; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 600; }
    .badge-low    { background: #2A9D8F; color: white; padding: 2px 8px; border-radius: 99px; font-size: 0.72rem; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; font-size: 1.7rem !important; }
    .stButton > button {
        background: linear-gradient(135deg, #E63946, #c1121f);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stDownloadButton > button {
        background: #1A1D27;
        color: #2A9D8F;
        border: 1px solid #2A9D8F;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDataFrame { border-radius: 10px; }
    section[data-testid="stSidebar"] { background-color: #13151F; border-right: 1px solid #2A2D3E; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Header ──
    st.markdown('<div class="hero-title">🚚 Delivery Optimization System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Amazon Dataset · Greedy Load Balancing · 3-Agent Assignment</div>', unsafe_allow_html=True)
    st.divider()

    # ─────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        use_weighted = st.toggle("Priority-Weighted Balancing", value=False,
                                  help="Weights distances by priority so High-priority routes are preferred")
        st.caption("High=1×  Medium=1.5×  Low=2× multiplier on distance when enabled")
        st.divider()
        
        st.markdown("### 📏 Distance Settings")
        force_small = st.toggle("Force Small Local Distances", value=True, help="Ignores true map coordinates to simulate short city deliveries (2-35 km).")
        st.divider()
        
        st.markdown("### 🏭 Warehouse Location")
        location_type = st.radio("Input Method", ["Address / City Text", "Latitude / Longitude"])
        
        if location_type == "Address / City Text":
            city = st.text_input("Enter City or Address (e.g., 'Andheri, Mumbai'):", "Bangalore")
            if city:
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
                    headers = {"User-Agent": "DeliveryApp/1.0"}
                    resp = requests.get(url, headers=headers, timeout=5).json()
                    if resp:
                        w_lat = float(resp[0]["lat"])
                        w_lon = float(resp[0]["lon"])
                        st.success(f"📍 Set map to: {resp[0]['display_name']}")
                    else:
                        st.error("Location not found.")
                        w_lat, w_lon = WAREHOUSE_LAT, WAREHOUSE_LON
                except Exception:
                    st.error("Network error fetching map location.", icon="⚠️")
                    w_lat, w_lon = WAREHOUSE_LAT, WAREHOUSE_LON
            else:
                w_lat, w_lon = WAREHOUSE_LAT, WAREHOUSE_LON
        else:
            w_lat = st.number_input("Latitude",  value=WAREHOUSE_LAT, format="%.4f")
            w_lon = st.number_input("Longitude", value=WAREHOUSE_LON, format="%.4f")
            
        st.divider()
        st.markdown("### 📂 Upload Data")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        use_sample = st.checkbox("Use bundled sample CSV", value=True)

    # ─────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────
    WAREHOUSE_LAT, WAREHOUSE_LON = w_lat, w_lon

    source = None
    if uploaded:
        source = uploaded
        st.success(f"✅ Loaded uploaded file: **{uploaded.name}**")
    elif use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "amazon_deliveries.csv")
        if os.path.exists(sample_path):
            source = sample_path
            st.info("📦 Using bundled sample dataset (50 deliveries)")

    if source is None:
        st.warning("⬆️ Upload a CSV file or enable the sample dataset to begin.")
        st.stop()

    # ── Read ──
    try:
        df_raw = read_csv(source, force_small=force_small)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    # ─────────────────────────────────────────────────────
    # SECTION 1 — RAW DATA PREVIEW
    # ─────────────────────────────────────────────────────
    with st.expander("📋 Raw Data Preview", expanded=False):
        st.dataframe(df_raw, use_container_width=True, height=280)

    # ─────────────────────────────────────────────────────
    # SECTION 2 — SORTED DELIVERIES
    # ─────────────────────────────────────────────────────
    st.markdown("### 📊 Sorted Deliveries")
    df_sorted = sort_deliveries(df_raw)

    def color_priority(val):
        colors = {"High": "#4d0009", "Medium": "#3d3000", "Low": "#003d38"}
        return f"background-color: {colors.get(val,'')}"

    cols_show = ["Location ID", "Delivery Priority", "Distance from warehouse"]
    extra = [c for c in df_sorted.columns if c not in cols_show]
    show_cols = cols_show + extra[:3]

    styled = df_sorted[show_cols].style.applymap(color_priority, subset=["Delivery Priority"])
    st.dataframe(styled, use_container_width=True, height=300)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Deliveries",     len(df_sorted))
    c2.metric("High Priority",  int((df_sorted["Delivery Priority"]=="High").sum()))
    c3.metric("Avg Distance (km)", f"{df_sorted['Distance from warehouse'].mean():.1f}")

    st.divider()

    # ─────────────────────────────────────────────────────
    # SECTION 3 — RUN OPTIMIZATION
    # ─────────────────────────────────────────────────────
    st.markdown("### 🔄 Assignment Optimization")

    if st.button("⚡ Run Optimization & Assign Deliveries"):
        with st.spinner("Optimizing…"):
            assignment = assign_deliveries(df_sorted, use_weighted=use_weighted)
        st.session_state["assignment"] = assignment
        st.session_state["df_sorted"]  = df_sorted
        st.success("✅ Optimization complete!")

    if "assignment" not in st.session_state:
        st.caption("Click the button above to run the greedy assignment algorithm.")
        st.stop()

    assignment = st.session_state["assignment"]
    df_sorted  = st.session_state["df_sorted"]

    # ─────────────────────────────────────────────────────
    # SECTION 4 — AGENT CARDS
    # ─────────────────────────────────────────────────────
    st.markdown("### 👤 Agent Assignments")
    card_classes = ["agent-1", "agent-2", "agent-3"]
    icons        = ["🔴", "🟢", "🟡"]

    cols = st.columns(3)
    for i, (agent, data) in enumerate(assignment.items()):
        with cols[i]:
            pri_counts = {p: data["priorities"].count(p) for p in ["High","Medium","Low"]}
            st.markdown(f"""
            <div class="agent-card {card_classes[i]}">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.4rem">
                    {icons[i]} {agent}
                </div>
                <div style="font-size:1.8rem;font-weight:800;font-family:'Syne',sans-serif">
                    {data['total']:.1f} km
                </div>
                <div style="color:#888;font-size:0.8rem;margin-top:0.2rem">{len(data['locations'])} deliveries</div>
                <hr style="border-color:#333;margin:0.6rem 0">
                <div style="font-size:0.78rem;display:flex;gap:6px;flex-wrap:wrap">
                    <span class="badge-high">High: {pri_counts['High']}</span>
                    <span class="badge-medium">Med: {pri_counts['Medium']}</span>
                    <span class="badge-low">Low: {pri_counts['Low']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 5 — VISUALISATION
    # ─────────────────────────────────────────────────────
    st.markdown("### 📈 Visualisation")
    fig = visualize_results(assignment)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ─────────────────────────────────────────────────────
    # SECTION 6 — DETAILED PLAN TABLE
    # ─────────────────────────────────────────────────────
    st.markdown("### 🗂️ Detailed Delivery Plan")
    plan_df = export_plan(assignment)
    st.dataframe(plan_df, use_container_width=True)

    # Per-agent breakdown
    st.markdown("#### Per-Agent Breakdown")
    agent_tabs = st.tabs(list(assignment.keys()))
    for tab, (agent, data) in zip(agent_tabs, assignment.items()):
        with tab:
            breakdown = pd.DataFrame({
                "Location ID":        data["locations"],
                "Distance (km)":      data["distances"],
                "Delivery Priority":  data["priorities"],
            })
            st.dataframe(breakdown, use_container_width=True, height=250)

    # ─────────────────────────────────────────────────────
    # SECTION 7 — EXPORT
    # ─────────────────────────────────────────────────────
    st.markdown("### 💾 Export")
    ex1, ex2 = st.columns(2)

    with ex1:
        csv_bytes = plan_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Delivery Plan CSV",
            data=csv_bytes,
            file_name=f"delivery_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with ex2:
        # Full per-delivery CSV
        rows = []
        for agent, data in assignment.items():
            for loc, dist, pri in zip(data["locations"], data["distances"], data["priorities"]):
                rows.append({"Agent": agent, "Location ID": loc,
                             "Distance (km)": dist, "Priority": pri})
        full_df  = pd.DataFrame(rows)
        full_csv = full_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Assignment CSV",
            data=full_csv,
            file_name=f"full_assignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────
    st.divider()
    balance = max(d["total"] for d in assignment.values()) - min(d["total"] for d in assignment.values())
    st.caption(f"⚖️ Load imbalance (max − min distance): **{balance:.2f} km** | "
               f"Algorithm: {'Priority-Weighted Greedy' if use_weighted else 'Standard Greedy'}")


if __name__ == "__main__":
    main()
