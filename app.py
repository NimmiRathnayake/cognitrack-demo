import streamlit as st, sqlite3, time, uuid, json, os
import pandas as pd, joblib
from datetime import datetime

# ------------------------------------------------------------------
# INSTRUMENTS
# NCS-6: Coelho et al. (2018), Assessment 27(8), appendix. Items 3 and 4
# reverse scored. 1 = extremely uncharacteristic ... 5 = extremely characteristic.
# ------------------------------------------------------------------
NCS6_ITEMS = [
    ("ncs1", "I would prefer complex to simple problems.", False),
    ("ncs2", "I like to have the responsibility of handling a situation that requires a lot of thinking.", False),
    ("ncs3", "Thinking is not my idea of fun.", True),
    ("ncs4", "I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.", True),
    ("ncs5", "I really enjoy a task that involves coming up with new solutions to problems.", False),
    ("ncs6", "I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.", False),
]
NCS_LABELS = ["1 - Extremely uncharacteristic of me", "2", "3", "4", "5 - Extremely characteristic of me"]

# GADS: PASTE YOUR FINALISED 11 ITEMS HERE: (id, text, component)
GADS_ITEMS = [
    ("gads1", "Using AI tools is the first thing I do when I face an academic problem.", "Salience"),
    ("gads2", "I feel anxious or frustrated when I cannot access AI tools during study.", "Withdrawal"),
    ("gads3", "I have tried to complete work without AI but found myself going back to it.", "Relapse"),
    # TODO: add the remaining 8 finalised items exactly as in your instrument document
]
GADS_FULL_LENGTH = 11   # the model was built on an 11-item total (range 11-55)
GADS_LABELS = ["1 - Never", "2 - Rarely", "3 - Sometimes", "4 - Often", "5 - Always"]

# PLACEHOLDER task battery (demo only; Section C is being rebuilt).
# (id, bloom-ish level, question, options, index of correct option)
TASKS = [
    ("t1", "Application",
     "A shop gives 20% off, then a further 10% off the reduced price. What is the total discount on a Rs. 1000 item?",
     ["30%", "28%", "25%", "18%"], 1),
    ("t2", "Analysis",
     "Argument: 'All indexed tables have a primary key. Table X has a primary key. So Table X is indexed.' What is wrong?",
     ["Nothing, it is valid", "It assumes the converse of the rule is true", "The rule is false for all tables", "It uses too many terms"], 1),
    ("t3", "Application",
     "What comes next in the sequence 2, 6, 12, 20, 30, ... ?",
     ["38", "40", "42", "44"], 2),
]

FACULTIES = ["IT / Computer Science", "Business / Management", "Engineering", "Other"]
FAC_SHORT = {"IT / Computer Science": "IT/CS", "Business / Management": "Business/Management",
             "Engineering": "Engineering", "Other": "Other"}

# Feedback per cluster (illustrative demo text, not validated interventions)
FEEDBACK = {
    "Independent thinkers": (
        "Your pattern suggests you usually think a problem through yourself and use AI selectively.",
        ["Keep attempting problems first, then use AI to check or extend your answer.",
         "Try explaining a concept in your own words after using AI, to keep the habit of self-checking."]),
    "Moderate AI users": (
        "Your pattern suggests regular AI use alongside your own thinking.",
        ["Set a rule: write your own first attempt (even a rough one) before opening an AI tool.",
         "Once a week, solve one study problem with AI switched off and note how it felt."]),
    "High AI dependency": (
        "Your pattern suggests heavy reliance on AI tools for study tasks.",
        ["Start small: choose one routine task per day to complete without AI.",
         "When you use AI, ask for hints or checks rather than full answers, then rewrite in your own words.",
         "Consider talking to your lecturer or a study-skills advisor about building independent study routines."]),
}


# ------------------------------------------------------------------
# SCORING (pure functions)
# ------------------------------------------------------------------
def score_ncs6(a):
    return sum((6 - a[i]) if rev else a[i] for i, _, rev in NCS6_ITEMS)

def score_gads_raw(a):
    return sum(a[i] for i, _, _ in GADS_ITEMS)

def gads_model_equivalent(a):
    """Scale to an 11-item total so it matches the model's training range.
    With the full 11 items this equals the raw sum. TEMPORARY while items are missing."""
    n = len(GADS_ITEMS)
    return round(score_gads_raw(a) * GADS_FULL_LENGTH / n, 1)

def gads_by_component(a):
    comp = {}
    for i, _, c in GADS_ITEMS:
        comp.setdefault(c, []).append(a[i])
    return {c: round(sum(v) / len(v), 2) for c, v in comp.items()}

def score_tasks(a):
    correct = sum(1 for tid, _, _, _, ans in TASKS if a.get(tid) == ans)
    return round(100 * correct / len(TASKS), 1)

def assign_cluster(ncs, gads, hours, task):
    if not os.path.exists("cluster_model.joblib"):
        return None
    m = joblib.load("cluster_model.joblib")
    X = pd.DataFrame([[ncs, gads, hours, task]], columns=m["features"])
    raw = int(m["model"].predict(m["scaler"].transform(X))[0])
    return m["label_map"][raw]


# ------------------------------------------------------------------
# DATABASE + SESSION STATE
# ------------------------------------------------------------------
conn = sqlite3.connect("cognitrack.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS responses
  (pid TEXT, started_at TEXT, finished_at TEXT, duration_sec REAL, page_times TEXT,
   answers TEXT, faculty TEXT, ai_hours REAL, ncs6 REAL, gads REAL,
   task_score REAL, cluster TEXT)""")

ss = st.session_state
ss.setdefault("pid", str(uuid.uuid4())[:8])
ss.setdefault("page", "consent")
ss.setdefault("page_times", {})
ss.setdefault("answers", {})
ss.setdefault("profile", {})

def go(nxt):
    now = time.time()
    ss.page_times[ss.page] = round(now - ss.get("page_entered", now), 1)
    ss.page, ss.page_entered = nxt, now
    st.rerun()

def likert_block(items, labels, prefix):
    out = {}
    for it in items:
        out[it[0]] = st.radio(it[1], options=[1, 2, 3, 4, 5], format_func=lambda x: labels[x - 1],
                              index=None, horizontal=True, key=f"{prefix}_{it[0]}")
        st.divider()
    return out


st.title("CogniTrack (Demo)")
st.caption("Research prototype. Results are illustrative and are not a diagnosis.")

# ---------------- consent ----------------
if ss.page == "consent":
    st.write("Demo only. No real participant data is collected.")
    ok = st.checkbox("I consent to take part")
    if st.button("Start") and ok:
        ss.t_start = time.time()
        ss.started_at = datetime.now().isoformat()
        ss.page_entered = ss.t_start
        go("profile")

# ---------------- profile ----------------
elif ss.page == "profile":
    st.subheader("About you")
    fac = st.selectbox("Faculty", FACULTIES, index=None, placeholder="Select your faculty")
    hours = st.number_input("On an average study day, about how many hours do you use generative AI tools?",
                            min_value=0.0, max_value=12.0, step=0.5, value=None)
    if st.button("Next"):
        if fac is None or hours is None:
            st.warning("Please answer both questions.")
        else:
            ss.profile = {"faculty": fac, "ai_hours": float(hours)}
            go("ncs")

# ---------------- NCS-6 ----------------
elif ss.page == "ncs":
    st.subheader("Section B1: Thinking preferences")
    st.caption("How characteristic is each statement of you?")
    resp = likert_block(NCS6_ITEMS, NCS_LABELS, "ncs")
    if st.button("Next"):
        if any(v is None for v in resp.values()):
            st.warning("Please answer every item.")
        else:
            ss.answers.update(resp); go("gads")

# ---------------- GADS ----------------
elif ss.page == "gads":
    st.subheader("Section B2: AI tool use")
    st.caption("How often does each statement apply to you?")
    resp = likert_block(GADS_ITEMS, GADS_LABELS, "gads")
    if st.button("Next"):
        if any(v is None for v in resp.values()):
            st.warning("Please answer every item.")
        else:
            ss.answers.update(resp); go("tasks")

# ---------------- placeholder task battery ----------------
elif ss.page == "tasks":
    st.subheader("Section C: Problem-solving tasks (demo placeholder)")
    st.caption("Please do not use AI tools or search engines for these tasks.")
    sel = {}
    for tid, lvl, q, opts, _ in TASKS:
        sel[tid] = st.radio(f"[{lvl}] {q}", options=list(range(len(opts))),
                            format_func=lambda i, o=opts: o[i], index=None, key=f"task_{tid}")
        st.divider()
    if st.button("Submit"):
        if any(v is None for v in sel.values()):
            st.warning("Please answer every task.")
        else:
            ss.answers.update(sel); go("done")

# ---------------- results ----------------
elif ss.page == "done":
    if "saved" not in ss:
        duration = round(time.time() - ss.t_start, 1)
        ncs = score_ncs6(ss.answers)
        gads_eq = gads_model_equivalent(ss.answers)
        task = score_tasks(ss.answers)
        cluster = assign_cluster(ncs, gads_eq, ss.profile["ai_hours"], task)
        conn.execute("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (ss.pid, ss.started_at, datetime.now().isoformat(), duration,
                      json.dumps(ss.page_times), json.dumps(ss.answers), ss.profile["faculty"],
                      ss.profile["ai_hours"], ncs, gads_eq, task, cluster))
        conn.commit()
        ss.saved, ss.duration, ss.ncs, ss.gads, ss.task, ss.cluster = True, duration, ncs, gads_eq, task, cluster

    st.success("Submitted. Thank you!")
    st.write(f"Completion time: {ss.duration} seconds")
    st.write("Time per section (s):", ss.page_times)

    c1, c2, c3 = st.columns(3)
    c1.metric("NCS-6", f"{ss.ncs} / 30")
    c2.metric("GADS (11-item equiv.)", f"{ss.gads} / 55")
    c3.metric("Task score", f"{ss.task} / 100")
    st.write("GADS by component (mean 1-5):", gads_by_component(ss.answers))

    st.subheader("Your profile group")
    if ss.cluster is None:
        st.warning("cluster_model.joblib not found. Run build_model.py first.")
    else:
        msg, tips = FEEDBACK[ss.cluster]
        st.info(f"**{ss.cluster}**: {msg}")
        st.write("Suggested next steps:")
        for t in tips:
            st.write(f"- {t}")
        st.caption("Groups come from a model fitted on SIMULATED data for this demo. They are not validated.")