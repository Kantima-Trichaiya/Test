import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Busy Buffet Dashboard", layout="wide")

st.title("Busy Buffet Dashboard")
st.caption("Task 1, Task 2, and Task 3 Analysis")

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
# Upload file
# -----------------------------
uploaded_file = st.file_uploader("Upload Busy Buffet Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload the Excel dataset to start.")
    st.stop()

df = clean_and_prepare_data(uploaded_file)
heat_df = build_heat_df(df)

# -----------------------------
# Overview
# -----------------------------
st.header("Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Groups", int(df["service_no"].count()))
c2.metric("Total Pax", int(df["pax"].sum()))
c3.metric("Waited Groups", int(df["queue_start_td"].notna().sum()))
c4.metric("Walk-away", int(df["walk_away"].sum()))
c5.metric("Avg Waiting", f"{df['waiting_time'].mean():.1f} min")
c6.metric("Avg Dining", f"{df['dining_time'].mean():.1f} min")

with st.expander("Show cleaned data"):
    st.dataframe(df)

# =========================================================
# TASK 1
# =========================================================
st.header("Task 1: Are the staff comments true?")

# -----------------------------
# Comment 1
# -----------------------------
st.subheader("Comment 1: In-house wait / Walk-in wait long and leave")

comment1 = df.groupby("Guest_type").agg(
    total_groups=("service_no", "count"),
    waited_groups=("queue_start_td", lambda s: s.notna().sum()),
    avg_wait=("waiting_time", "mean"),
    median_wait=("waiting_time", "median"),
    walkaway_count=("walk_away", "sum")
)

comment1["wait_rate"] = comment1["waited_groups"] / comment1["total_groups"]
comment1["walkaway_rate_among_waited"] = comment1["walkaway_count"] / comment1["waited_groups"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
comment1["avg_wait"].plot(kind="bar", ax=ax[0], title="Avg Waiting Time by Guest Type")
comment1["wait_rate"].plot(kind="bar", ax=ax[1], title="Wait Rate by Guest Type")
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

st.dataframe(comment1.round(3))
st.markdown("""
**Judgement: Partially true**

- In-house guests do wait for tables.
- Walk-in guests wait longer on average.
- However, walk-in guests are not more likely to walk away than in-house guests based on this dataset.
""")

# -----------------------------
# Comment 2
# -----------------------------
st.subheader("Comment 2: We are very busy every day of the week")

comment2 = df.groupby("date").agg(
    total_groups=("service_no", "count"),
    total_pax=("pax", "sum"),
    waited_groups=("queue_start_td", lambda s: s.notna().sum()),
    avg_wait=("waiting_time", "mean"),
    walkaway_count=("walk_away", "sum")
)

comment2["wait_rate"] = comment2["waited_groups"] / comment2["total_groups"]

fig, ax = plt.subplots(1, 4, figsize=(22, 4))
comment2["total_groups"].plot(kind="bar", ax=ax[0], title="Total Groups by Day ID")
comment2["total_pax"].plot(kind="bar", ax=ax[1], title="Total Pax by Day ID")
comment2["waited_groups"].plot(kind="bar", ax=ax[2], title="Waited Groups by Day ID")
comment2["avg_wait"].plot(kind="bar", ax=ax[3], title="Avg Waiting Time by Day ID")

for i, v in enumerate(comment2["total_groups"]):
    ax[0].text(i, v + 0.5, f"{int(v)}", ha="center", fontsize=8)

for i, v in enumerate(comment2["total_pax"]):
    if pd.notna(v):
        ax[1].text(i, v + 1, f"{int(v)}", ha="center", fontsize=8)

for i, v in enumerate(comment2["waited_groups"]):
    ax[2].text(i, v + 0.5, f"{int(v)}", ha="center", fontsize=8)

for i, v in enumerate(comment2["avg_wait"]):
    if pd.notna(v):
        ax[3].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

ax[0].set_ylabel("Groups")
ax[1].set_ylabel("Pax")
ax[2].set_ylabel("Groups")
ax[3].set_ylabel("Minutes")
plt.tight_layout()
st.pyplot(fig)

comment2_reset = comment2.reset_index().sort_values("date")
fig, ax = plt.subplots(1, 2, figsize=(14, 4))

ax[0].plot(comment2_reset["date"].astype(str), comment2_reset["total_pax"], marker="o")
ax[0].set_title("Total Pax by Day ID")
ax[0].set_xlabel("Day ID")
ax[0].set_ylabel("Pax")
for i, v in enumerate(comment2_reset["total_pax"]):
    ax[0].text(i, v + 1, f"{int(v)}", ha="center")

ax[1].plot(comment2_reset["date"].astype(str), comment2_reset["waited_groups"], marker="o")
ax[1].set_title("Waited Groups by Day ID")
ax[1].set_xlabel("Day ID")
ax[1].set_ylabel("Groups")
for i, v in enumerate(comment2_reset["waited_groups"]):
    ax[1].text(i, v + 0.5, f"{int(v)}", ha="center")

plt.tight_layout()
st.pyplot(fig)

st.dataframe(comment2.round(3))
st.markdown("""
**Judgement: Partially true**

- The buffet has customers on every recorded day.
- However, severe congestion happens only on some days, especially day 143 and day 153.
- Therefore, the buffet is active every day, but not equally overloaded every day.
""")

# -----------------------------
# Comment 3
# -----------------------------
st.subheader("Comment 3: Walk-in customers sit the whole day")

comment3 = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].groupby("Guest_type").agg(
    seated_groups=("service_no", "count"),
    avg_dining=("dining_time", "mean"),
    median_dining=("dining_time", "median"),
    max_dining=("dining_time", "max")
)

long_stay = df[
    df["meal_start_td"].notna() &
    df["meal_end_td"].notna() &
    (df["dining_time"] > 90)
].groupby("Guest_type")["service_no"].count()

comment3["long_stay_over_90min"] = long_stay
comment3["long_stay_rate_over_90min"] = comment3["long_stay_over_90min"] / comment3["seated_groups"]
comment3 = comment3.fillna(0)

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
comment3["avg_dining"].plot(kind="bar", ax=ax[0], title="Average Dining Time by Guest Type")
comment3["median_dining"].plot(kind="bar", ax=ax[1], title="Median Dining Time by Guest Type")
comment3["long_stay_rate_over_90min"].plot(kind="bar", ax=ax[2], title="Long-stay Rate (> 90 min)")

for i, v in enumerate(comment3["avg_dining"]):
    ax[0].text(i, v + 1, f"{v:.1f}", ha="center")

for i, v in enumerate(comment3["median_dining"]):
    ax[1].text(i, v + 1, f"{v:.1f}", ha="center")

for i, v in enumerate(comment3["long_stay_rate_over_90min"]):
    ax[2].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

ax[0].set_ylabel("Minutes")
ax[1].set_ylabel("Minutes")
ax[2].set_ylabel("Rate")
plt.tight_layout()
st.pyplot(fig)

seated_only = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].copy()

fig, ax = plt.subplots(figsize=(8, 4))
for guest_type in seated_only["Guest_type"].dropna().unique():
    subset = seated_only[seated_only["Guest_type"] == guest_type]["dining_time"].dropna()
    ax.hist(subset, bins=15, alpha=0.5, label=guest_type)
ax.set_title("Dining Time Distribution by Guest Type")
ax.set_xlabel("Dining Time (minutes)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 4))
seated_only.boxplot(column="dining_time", by="Guest_type", ax=ax)
ax.set_title("Dining Time Distribution by Guest Type")
ax.set_xlabel("Guest Type")
ax.set_ylabel("Minutes")
plt.suptitle("")
st.pyplot(fig)

st.dataframe(comment3.round(3))
st.markdown("""
**Judgement: Partially true**

- Walk-in guests do stay longer than in-house guests.
- Walk-in also has a much higher long-stay rate.
- However, the phrase "sit the whole day" is exaggerated.
""")

# =========================================================
# TASK 2
# =========================================================
st.header("Task 2: Why the proposed actions do not work")

# -----------------------------
# Action 1
# -----------------------------
st.subheader("Action 1: Reduce seating time (5 hours to less)")

action1 = df[df["meal_start_td"].notna() & df["meal_end_td"].notna()].copy()

summary_a1 = pd.Series({
    "seated_groups": len(action1),
    "avg_dining": action1["dining_time"].mean(),
    "median_dining": action1["dining_time"].median(),
    "p90_dining": action1["dining_time"].quantile(0.90),
    "p95_dining": action1["dining_time"].quantile(0.95),
    "over_120min": (action1["dining_time"] > 120).sum(),
    "over_180min": (action1["dining_time"] > 180).sum(),
    "over_240min": (action1["dining_time"] > 240).sum(),
    "over_300min": (action1["dining_time"] > 300).sum()
})

st.write("Action 1 Summary")
st.dataframe(summary_a1.round(2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(action1["dining_time"].dropna(), bins=20, alpha=0.7)
ax.axvline(120, linestyle="--", label="120 min")
ax.axvline(180, linestyle="--", label="180 min")
ax.axvline(300, linestyle="--", label="300 min (5 hours)")
ax.set_title("Dining Time Distribution")
ax.set_xlabel("Dining Time (minutes)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

st.markdown("""
**Conclusion:** Reducing the seating-time limit is not effective because almost all customers dine far below 5 hours.
""")

# -----------------------------
# Action 2
# -----------------------------
st.subheader("Action 2: Increase price every day to 259")

action2 = df.groupby("date").agg(
    total_groups=("service_no", "count"),
    total_pax=("pax", "sum"),
    waited_groups=("queue_start_td", lambda s: s.notna().sum()),
    avg_wait=("waiting_time", "mean"),
    walkaway_count=("walk_away", "sum")
)

action2["wait_rate"] = action2["waited_groups"] / action2["total_groups"]

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
action2["total_pax"].plot(kind="bar", ax=ax[0], title="Total Pax by Day ID")
action2["waited_groups"].plot(kind="bar", ax=ax[1], title="Waited Groups by Day ID")
action2["avg_wait"].plot(kind="bar", ax=ax[2], title="Avg Waiting Time by Day ID")

for i, v in enumerate(action2["total_pax"]):
    if pd.notna(v):
        ax[0].text(i, v + 1, f"{int(v)}", ha="center", fontsize=8)

for i, v in enumerate(action2["waited_groups"]):
    ax[1].text(i, v + 0.5, f"{int(v)}", ha="center", fontsize=8)

for i, v in enumerate(action2["avg_wait"]):
    if pd.notna(v):
        ax[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

ax[0].set_ylabel("Pax")
ax[1].set_ylabel("Groups")
ax[2].set_ylabel("Minutes")
plt.tight_layout()
st.pyplot(fig)

st.dataframe(action2.round(3))
st.markdown("""
**Conclusion:** A uniform daily price increase is not targeted because severe congestion happens only on some days, not every day.
""")

# -----------------------------
# Action 3
# -----------------------------
st.subheader("Action 3: Queue skipping for in-house guests")

heat_tables = heat_df.pivot(index="date", columns="hour", values="active_tables").sort_index()

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(heat_tables, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
ax.set_title("Active Tables by Day ID and Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Day ID")
st.pyplot(fig)

action3 = df.groupby("Guest_type").agg(
    total_groups=("service_no", "count"),
    waited_groups=("queue_start_td", lambda s: s.notna().sum()),
    avg_wait=("waiting_time", "mean"),
    walkaway_count=("walk_away", "sum"),
    seated_groups=("meal_start_td", lambda s: s.notna().sum())
)

action3["wait_rate"] = action3["waited_groups"] / action3["total_groups"]
action3["walkaway_rate_among_waited"] = action3["walkaway_count"] / action3["waited_groups"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
action3["waited_groups"].plot(kind="bar", ax=ax[0], title="Waited Groups by Guest Type")
action3["avg_wait"].plot(kind="bar", ax=ax[1], title="Avg Waiting Time by Guest Type")
action3["walkaway_rate_among_waited"].plot(kind="bar", ax=ax[2], title="Walk-away Rate Among Waited")

for i, v in enumerate(action3["waited_groups"]):
    ax[0].text(i, v + 0.5, f"{int(v)}", ha="center")

for i, v in enumerate(action3["avg_wait"]):
    if pd.notna(v):
        ax[1].text(i, v + 1, f"{v:.1f}", ha="center")

for i, v in enumerate(action3["walkaway_rate_among_waited"]):
    if pd.notna(v):
        ax[2].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

ax[0].set_ylabel("Groups")
ax[1].set_ylabel("Minutes")
ax[2].set_ylabel("Rate")
plt.tight_layout()
st.pyplot(fig)

st.dataframe(action3.round(3))
st.markdown("""
**Conclusion:** Queue skipping does not increase capacity. It only shifts the waiting burden from one group to another.
""")

# =========================================================
# TASK 3
# =========================================================
st.header("Task 3: Recommended Action")

st.subheader("Recommended Action: Prioritize in-house guests during peak hours")

task3 = df.groupby("Guest_type").agg(
    total_groups=("service_no", "count"),
    waited_groups=("queue_start_td", lambda s: s.notna().sum()),
    avg_wait=("waiting_time", "mean"),
    median_wait=("waiting_time", "median"),
    walkaway_count=("walk_away", "sum"),
    seated_groups=("meal_start_td", lambda s: s.notna().sum())
)

task3["wait_rate"] = task3["waited_groups"] / task3["total_groups"]
task3["walkaway_rate_among_waited"] = task3["walkaway_count"] / task3["waited_groups"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
task3["wait_rate"].plot(kind="bar", ax=ax[0], title="Wait Rate by Guest Type")
task3["avg_wait"].plot(kind="bar", ax=ax[1], title="Average Waiting Time by Guest Type")
task3["walkaway_rate_among_waited"].plot(kind="bar", ax=ax[2], title="Walk-away Rate Among Waited")

for i, v in enumerate(task3["wait_rate"]):
    if pd.notna(v):
        ax[0].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

for i, v in enumerate(task3["avg_wait"]):
    if pd.notna(v):
        ax[1].text(i, v + 1, f"{v:.1f}", ha="center")

for i, v in enumerate(task3["walkaway_rate_among_waited"]):
    if pd.notna(v):
        ax[2].text(i, v + 0.01, f"{v*100:.1f}%", ha="center")

ax[0].set_ylabel("Rate")
ax[1].set_ylabel("Minutes")
ax[2].set_ylabel("Rate")
plt.tight_layout()
st.pyplot(fig)

peak_hours = heat_df.groupby("hour").agg(
    avg_active_tables=("active_tables", "mean"),
    avg_active_pax=("active_pax", "mean")
).reset_index()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(peak_hours["hour"], peak_hours["avg_active_tables"], marker="o")
ax.set_title("Average Active Tables by Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Average Active Tables")
st.pyplot(fig)

st.dataframe(task3.round(3))
st.markdown("""
**Recommendation**

A better strategy is to reserve or prioritize some tables for in-house guests during peak hours only.

- In-house guests have a higher walk-away rate once they wait.
- The data shows that table pressure is concentrated in peak hours.
- Therefore, a targeted peak-hour priority policy is more practical than a full-day queue-skipping rule.
""")
