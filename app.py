import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Busy Buffet Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #0c2232;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }

    .hero {
        background: linear-gradient(135deg, #0f2738 0%, #142b3c 100%);
        border: 1px solid #5e6c79;
        border-radius: 22px;
        padding: 26px 28px;
        color: #e4eaed;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        color: #e4eaed;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin-top: 8px;
        font-size: 1rem;
        color: #c7d1d7;
    }

    .section-card {
        background: #142b3c;
        border: 1px solid #5e6c79;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
        margin-bottom: 18px;
    }

    div[data-testid="stMetric"] {
        background: #142b3c;
        border: 1px solid #5e6c79;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.16);
    }

    div[data-testid="stMetricLabel"] {
        color: #b7c3cb;
    }

    div[data-testid="stMetricValue"] {
        color: #e4eaed;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #5e6c79;
        border-radius: 14px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #e4eaed;
    }

    p, label, .small-note {
        color: #c7d1d7;
    }

    .judgement-box {
        background: #0f2738;
        border-left: 6px solid #5e6c79;
        padding: 12px 16px;
        border-radius: 12px;
        margin-top: 10px;
        color: #e4eaed;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🍽️ Busy Buffet Dashboard</h1>
    <p>Customer waiting, dining duration, table pressure, and recommendation analysis</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Chart style
# -----------------------------
def apply_dark_chart_style():
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = "#142b3c"
    plt.rcParams["axes.facecolor"] = "#142b3c"
    plt.rcParams["axes.edgecolor"] = "#5e6c79"
    plt.rcParams["axes.labelcolor"] = "#e4eaed"
    plt.rcParams["xtick.color"] = "#e4eaed"
    plt.rcParams["ytick.color"] = "#e4eaed"
    plt.rcParams["text.color"] = "#e4eaed"
    plt.rcParams["axes.titlecolor"] = "#e4eaed"
    plt.rcParams["grid.color"] = "#5e6c79"

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def clean_and_prepare_data(file):
    all_sheets = pd.read_excel(file, sheet_name=None, dtype=str)
    df_list = []

    for name, sheet in all_sheets.items():
        sheet = sheet.loc[:, ~sheet.columns.str.contains("^Unnamed", na=False)].copy()
        sheet["date"] = str(name)
        df_list.append(sheet)

    df = pd.concat(df_list, ignore_index=True)

    df = df.rename(columns={
        "service_no.": "service_no",
        "table_no.": "table_no"
    })

    if "table_no" not in df.columns:
        df["table_no"] = None

    time_cols = ["queue_start", "queue_end", "meal_start", "meal_end"]

    for col in time_cols:
        if col not in df.columns:
            df[col] = None

        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({
            "nan": None,
            "NaN": None,
            "None": None,
            "NaT": None,
            "": None
        })
        df[col + "_td"] = pd.to_timedelta(df[col], errors="coerce")

    df["pax"] = pd.to_numeric(df["pax"], errors="coerce")

    df["waiting_time"] = (df["queue_end_td"] - df["queue_start_td"]).dt.total_seconds() / 60
    df["dining_time"] = (df["meal_end_td"] - df["meal_start_td"]).dt.total_seconds() / 60
    df["walk_away"] = df["queue_start_td"].notna() & df["meal_start_td"].isna()

    df.loc[df["waiting_time"] < 0, "waiting_time"] = np.nan
    df.loc[df["dining_time"] < 0, "dining_time"] = np.nan

    return df


def split_tables(table_no):
    if pd.isna(table_no):
        return []
    table_no = str(table_no).strip()
    if table_no in ["", "nan", "NaN", "None", "99"]:
        return []
    return [x.strip() for x in table_no.split("-") if x.strip()]


@st.cache_data
def build_heat_df(df):
    seated = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].copy()
    rows = []

    for day_id, g in seated.groupby("date"):
        for h in range(6, 15):
            slot_start = pd.to_timedelta(f"{h:02d}:00:00")
            slot_end = pd.to_timedelta(f"{h+1:02d}:00:00")

            active = g[
                (g["meal_start_td"] < slot_end) &
                (g["meal_end_td"] > slot_start)
            ].copy()

            active_table_units = []
            for t in active["table_no"]:
                active_table_units.extend(split_tables(t))

            rows.append({
                "date": day_id,
                "hour": f"{h:02d}:00",
                "active_tables": len(set(active_table_units)),
                "active_groups": len(active),
                "active_pax": active["pax"].sum()
            })

    return pd.DataFrame(rows)

# -----------------------------
# Section renderers
# -----------------------------
def show_comment1(df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Comment 1")
    st.caption("In-house wait / Walk-in wait long and leave")

    comment1 = df.groupby("Guest_type").agg(
        total_groups=("service_no", "count"),
        waited_groups=("queue_start_td", lambda s: s.notna().sum()),
        avg_wait=("waiting_time", "mean"),
        walkaway_count=("walk_away", "sum")
    )
    comment1["wait_rate"] = comment1["waited_groups"] / comment1["total_groups"]
    comment1["walkaway_rate_among_waited"] = comment1["walkaway_count"] / comment1["waited_groups"]

    apply_dark_chart_style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    comment1["avg_wait"].plot(kind="bar", ax=ax[0], title="Avg Waiting Time")
    comment1["wait_rate"].plot(kind="bar", ax=ax[1], title="Wait Rate")
    comment1["walkaway_rate_among_waited"].plot(kind="bar", ax=ax[2], title="Walk-away Rate Among Waited")

    for i, v in enumerate(comment1["avg_wait"]):
        if pd.notna(v):
            ax[0].text(i, v + 1, f"{v:.1f}", ha="center")
    for i, v in enumerate(comment1["wait_rate"]):
        if pd.notna(v):
            ax[1].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")
    for i, v in enumerate(comment1["walkaway_rate_among_waited"]):
        if pd.notna(v):
            ax[2].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

    ax[0].set_ylabel("Minutes")
    ax[1].set_ylabel("Rate")
    ax[2].set_ylabel("Rate")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(comment1.round(3), use_container_width=True)
    st.markdown("""
    <div class="judgement-box">
        <b>Judgement: Partially true</b><br>
        In-house guests do wait for tables. Walk-in guests wait longer on average.
        However, walk-in guests are not more likely to walk away than in-house guests based on this dataset.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_comment2(df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Comment 2")
    st.caption("We are very busy every day of the week")

    comment2 = df.groupby("date").agg(
        total_groups=("service_no", "count"),
        total_pax=("pax", "sum"),
        waited_groups=("queue_start_td", lambda s: s.notna().sum()),
        avg_wait=("waiting_time", "mean"),
        walkaway_count=("walk_away", "sum")
    )

    apply_dark_chart_style()
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    comment2["total_pax"].plot(kind="bar", ax=ax[0], title="Total Pax by Day ID")
    comment2["waited_groups"].plot(kind="bar", ax=ax[1], title="Waited Groups by Day ID")
    comment2["avg_wait"].plot(kind="bar", ax=ax[2], title="Avg Waiting Time by Day ID")

    for i, v in enumerate(comment2["total_pax"]):
        if pd.notna(v):
            ax[0].text(i, v + 1, f"{int(v)}", ha="center", fontsize=8)
    for i, v in enumerate(comment2["waited_groups"]):
        ax[1].text(i, v + 0.5, f"{int(v)}", ha="center", fontsize=8)
    for i, v in enumerate(comment2["avg_wait"]):
        if pd.notna(v):
            ax[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    ax[0].set_ylabel("Pax")
    ax[1].set_ylabel("Groups")
    ax[2].set_ylabel("Minutes")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(comment2.round(3), use_container_width=True)
    st.markdown("""
    <div class="judgement-box">
        <b>Judgement: Partially true</b><br>
        The buffet has customers on every recorded day. However, severe congestion happens only on some days,
        especially day 143 and day 153.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_comment3(df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Comment 3")
    st.caption("Walk-in customers sit the whole day")

    comment3 = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].groupby("Guest_type").agg(
        seated_groups=("service_no", "count"),
        avg_dining=("dining_time", "mean"),
        median_dining=("dining_time", "median")
    )

    long_stay = df[
        df["meal_start_td"].notna() &
        df["meal_end_td"].notna() &
        (df["dining_time"] > 90)
    ].groupby("Guest_type")["service_no"].count()

    comment3["long_stay_over_90min"] = long_stay
    comment3["long_stay_rate_over_90min"] = comment3["long_stay_over_90min"] / comment3["seated_groups"]
    comment3 = comment3.fillna(0)

    apply_dark_chart_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    comment3["avg_dining"].plot(kind="bar", ax=ax[0], title="Average Dining Time")
    comment3["long_stay_rate_over_90min"].plot(kind="bar", ax=ax[1], title="Long-stay Rate (>90 min)")

    for i, v in enumerate(comment3["avg_dining"]):
        ax[0].text(i, v + 1, f"{v:.1f}", ha="center")
    for i, v in enumerate(comment3["long_stay_rate_over_90min"]):
        ax[1].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

    ax[0].set_ylabel("Minutes")
    ax[1].set_ylabel("Rate")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    apply_dark_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    seated_only = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].copy()
    seated_only.boxplot(column="dining_time", by="Guest_type", ax=ax)
    ax.set_title("Dining Time Distribution")
    ax.set_xlabel("Guest Type")
    ax.set_ylabel("Minutes")
    plt.suptitle("")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(comment3.round(3), use_container_width=True)
    st.markdown("""
    <div class="judgement-box">
        <b>Judgement: Partially true</b><br>
        Walk-in guests do stay longer than in-house guests, but the phrase
        “sit the whole day” is exaggerated.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_task2(df, heat_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Action 1: Reduce seating time")

    action1 = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].copy()

    apply_dark_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(action1["dining_time"].dropna(), bins=20, alpha=0.75)
    ax.axvline(300, linestyle="--", label="300 min (5 hours)")
    ax.set_title("Dining Time Distribution")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    <div class="judgement-box">
        <b>Conclusion:</b> Not effective. Most customers dine far below 5 hours.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Action 2: Increase price every day to 259")

    action2 = df.groupby("date").agg(
        total_pax=("pax", "sum"),
        waited_groups=("queue_start_td", lambda s: s.notna().sum()),
        avg_wait=("waiting_time", "mean")
    )

    apply_dark_chart_style()
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    action2["total_pax"].plot(kind="bar", ax=ax[0], title="Total Pax by Day ID")
    action2["waited_groups"].plot(kind="bar", ax=ax[1], title="Waited Groups by Day ID")
    action2["avg_wait"].plot(kind="bar", ax=ax[2], title="Avg Waiting Time by Day ID")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    <div class="judgement-box">
        <b>Conclusion:</b> Not targeted. Congestion happens only on some days.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Action 3: Queue skipping for in-house guests")

    heat_tables = heat_df.pivot(index="date", columns="hour", values="active_tables").sort_index()

    apply_dark_chart_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heat_tables, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Active Tables by Day ID and Hour")
    st.pyplot(fig)
    plt.close(fig)

    action3 = df.groupby("Guest_type").agg(
        waited_groups=("queue_start_td", lambda s: s.notna().sum()),
        avg_wait=("waiting_time", "mean"),
        walkaway_count=("walk_away", "sum")
    )

    st.dataframe(action3.round(3), use_container_width=True)
    st.markdown("""
    <div class="judgement-box">
        <b>Conclusion:</b> Queue skipping does not increase capacity. It only shifts the waiting burden.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_task3(df, heat_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Recommended Action")
    st.caption("Prioritize some tables for in-house guests during peak hours")

    task3 = df.groupby("Guest_type").agg(
        waited_groups=("queue_start_td", lambda s: s.notna().sum()),
        avg_wait=("waiting_time", "mean"),
        walkaway_count=("walk_away", "sum"),
        total_groups=("service_no", "count")
    )
    task3["wait_rate"] = task3["waited_groups"] / task3["total_groups"]
    task3["walkaway_rate_among_waited"] = task3["walkaway_count"] / task3["waited_groups"]

    apply_dark_chart_style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    task3["wait_rate"].plot(kind="bar", ax=ax[0], title="Wait Rate")
    task3["avg_wait"].plot(kind="bar", ax=ax[1], title="Avg Waiting Time")
    task3["walkaway_rate_among_waited"].plot(kind="bar", ax=ax[2], title="Walk-away Rate Among Waited")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    peak_hours = heat_df.groupby("hour").agg(
        avg_active_tables=("active_tables", "mean")
    ).reset_index()

    apply_dark_chart_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(peak_hours["hour"], peak_hours["avg_active_tables"], marker="o")
    ax.set_title("Average Active Tables by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Active Tables")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    <div class="judgement-box">
        <b>Recommendation:</b> Reserve or prioritize some tables for in-house guests during peak hours only.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload Busy Buffet Dataset (.xlsx)", type=["xlsx"])

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Task 1", "Task 2", "Task 3"]
)

if uploaded_file is None:
    st.info("Upload the Excel dataset from the sidebar to start.")
    st.stop()

df = clean_and_prepare_data(uploaded_file)
heat_df = build_heat_df(df)

# -----------------------------
# Pages
# -----------------------------
if page == "Overview":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Executive Summary")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Groups", int(df["service_no"].count()))
    c2.metric("Total Pax", int(df["pax"].sum()))
    c3.metric("Waited Groups", int(df["queue_start_td"].notna().sum()))
    c4.metric("Walk-away", int(df["walk_away"].sum()))
    c5.metric("Avg Waiting", f"{df['waiting_time'].mean():.1f} min")
    c6.metric("Avg Dining", f"{df['dining_time'].mean():.1f} min")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Peak-hour Table Pressure")

    heat_tables = heat_df.pivot(index="date", columns="hour", values="active_tables").sort_index()

    apply_dark_chart_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heat_tables, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("Active Tables by Day ID and Hour")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show cleaned data"):
        st.dataframe(df, use_container_width=True)

elif page == "Task 1":
    st.header("Task 1: Are the staff comments true?")
    show_comment1(df)
    show_comment2(df)
    show_comment3(df)

elif page == "Task 2":
    st.header("Task 2: Why the proposed actions do not work")
    show_task2(df, heat_df)

elif page == "Task 3":
    st.header("Task 3: Recommended Action")
    show_task3(df, heat_df)
