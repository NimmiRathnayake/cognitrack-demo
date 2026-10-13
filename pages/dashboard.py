import streamlit as st, sqlite3, os
import pandas as pd, joblib
import altair as alt
from scipy import stats

st.set_page_config(page_title="CogniTrack Researcher Dashboard", layout="wide")
st.title("CogniTrack: Researcher Dashboard (Demo)")

# ---- simple password gate (demo-level only, NOT real security) ----
PASSWORD = os.environ.get("DASH_PASSWORD", "demo123")
if "dash_ok" not in st.session_state:
    st.session_state.dash_ok = False
if not st.session_state.dash_ok:
    pw = st.text_input("Researcher password", type="password")
    if st.button("Enter"):
        if pw == PASSWORD:
            st.session_state.dash_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

FAC_SHORT = {"IT / Computer Science": "IT/CS", "Business / Management": "Business/Management",
             "Engineering": "Engineering", "Other": "Other"}
FEATURES = ["ncs6", "gads", "ai_hours", "task_score"]

@st.cache_data(ttl=5)
def load_live():
    if not os.path.exists("cognitrack.db"):
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM responses", sqlite3.connect("cognitrack.db"))
    if not df.empty:
        df["faculty"] = df["faculty"].map(FAC_SHORT).fillna(df["faculty"])
        df = df.rename(columns={"ncs6": "ncs6", "gads": "gads"})
    return df

@st.cache_data
def load_sim():
    if not (os.path.exists("synthetic_data.csv") and os.path.exists("cluster_model.joblib")):
        return pd.DataFrame()
    df = pd.read_csv("synthetic_data.csv")
    m = joblib.load("cluster_model.joblib")
    raw = m["model"].predict(m["scaler"].transform(df[m["features"]]))
    df["cluster"] = [m["label_map"][int(r)] for r in raw]
    return df

source = st.radio("Data source", ["Live demo responses", "Simulated dataset (360)"], horizontal=True)
df = load_live() if source.startswith("Live") else load_sim()
if source.startswith("Simulated"):
    st.warning("SIMULATED data generated to test the pipeline. Not real participants; clusters are not validated.")

if df.empty:
    st.info("No data yet. Complete the survey on the main page, or choose the simulated dataset.")
    st.stop()

# ---- headline numbers ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Participants", len(df))
c2.metric("Mean NCS-6", f"{df['ncs6'].mean():.1f} / 30")
c3.metric("Mean GADS", f"{df['gads'].mean():.1f} / 55")
c4.metric("Mean AI hours/day", f"{df['ai_hours'].mean():.1f}")

# ---- cluster distribution + profiles ----
st.subheader("Cluster distribution")
left, right = st.columns(2)
CLUSTER_ORDER = ["Independent thinkers", "Moderate AI users", "High AI dependency"]
CLUSTER_COLORS = ["#0F766E", "#D97706", "#BE123C"]
counts = df["cluster"].value_counts().reindex(CLUSTER_ORDER, fill_value=0)
chart_df = pd.DataFrame({"cluster": counts.index, "n": counts.values})
chart_df["label"] = [f"{n}  ({n / max(len(df), 1) * 100:.0f}%)" for n in chart_df["n"]]
base = alt.Chart(chart_df).encode(
    y=alt.Y("cluster:N", sort=CLUSTER_ORDER, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13)),
)
bars = base.mark_bar(size=34, cornerRadiusEnd=4).encode(
    x=alt.X("n:Q", title="Number of participants",
            scale=alt.Scale(domain=[0, max(int(chart_df["n"].max()) * 1.35, 2)]),
            axis=alt.Axis(tickMinStep=1, format="d")),
    color=alt.Color("cluster:N", scale=alt.Scale(domain=CLUSTER_ORDER, range=CLUSTER_COLORS), legend=None),
)
text = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold").encode(
    x="n:Q", text="label:N")
left.altair_chart((bars + text).properties(height=230), width="stretch")
right.dataframe(df.groupby("cluster")[FEATURES].mean().round(1), width='stretch')

# ---- faculty comparison ----
st.subheader("Faculty comparison")
l2, r2 = st.columns(2)
l2.dataframe(df.groupby("faculty")[FEATURES].mean().round(1), width='stretch')
r2.dataframe(pd.crosstab(df["faculty"], df["cluster"]), width='stretch')

fac_counts = df["faculty"].value_counts()
if {"IT/CS", "Business/Management"} <= set(fac_counts.index) and fac_counts[["IT/CS", "Business/Management"]].min() >= 3:
    a = df.loc[df.faculty == "IT/CS", "gads"]; b = df.loc[df.faculty == "Business/Management", "gads"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    st.write(f"Welch t-test, GADS, IT/CS vs Business/Management: t = {t:.2f}, p = {p:.3f}")
else:
    st.caption("t-test needs at least 3 participants in both IT/CS and Business/Management.")

# ---- correlations ----
st.subheader("Correlations (Pearson r)")
if len(df) >= 10:
    pairs = [("gads", "ai_hours"), ("gads", "task_score"), ("ncs6", "gads"), ("ncs6", "task_score")]
    rows = []
    for x, y in pairs:
        r, p = stats.pearsonr(df[x], df[y])
        rows.append({"pair": f"{x} vs {y}", "r": round(r, 2), "p": round(p, 4)})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
else:
    st.caption("Need at least 10 participants for correlations.")

# ---- Section C completion time (live data only) ----
st.subheader("Section C completion time")
if "section_c_sec" in df.columns:
    fast = st.slider("Flag as too fast if under (seconds)", 5, 300, 20)
    slow = st.slider("Flag as too slow if over (minutes)", 5, 120, 30)
    d = df.copy()
    d["time_flag"] = "ok"
    d.loc[d.section_c_sec < fast, "time_flag"] = "too fast"
    d.loc[d.section_c_sec > slow * 60, "time_flag"] = "too slow"
    t1, t2, t3 = st.columns(3)
    t1.metric("Median", f"{d.section_c_sec.median():.0f} s")
    t2.metric("Fastest", f"{d.section_c_sec.min():.0f} s")
    t3.metric("Slowest", f"{d.section_c_sec.max():.0f} s")
    st.write("Flag counts:", d["time_flag"].value_counts().to_dict())
    st.caption("Flags are screening aids only. A very fast time may mean careless answering; a very slow time may mean "
               "the participant left the page open or used outside help. Review flagged rows before excluding anyone.")
    st.dataframe(d[["pid", "faculty", "section_c_sec", "time_flag", "task_score", "cluster"]],
                 width='stretch', hide_index=True)
    df = d
else:
    st.caption("The simulated dataset has no timing data. Switch to Live demo responses.")

# ---- export ----
st.download_button("Download data as CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name="cognitrack_export.csv", mime="text/csv")
