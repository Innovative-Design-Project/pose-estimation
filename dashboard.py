"""Post-session analytics dashboard.

    streamlit run dashboard.py
    COACH_DB=/path/to/coach.db streamlit run dashboard.py

Reads sessions/coach.db written by live.py and analyze.py.
"""
import os

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from coach.coaching import TARGETS, metric_label
from coach.llm import session_summary
from coach.store import DEFAULT_DB, SessionStore

load_dotenv()  # ANTHROPIC_API_KEY / ELEVENLABS_API_KEY from .env

st.set_page_config(page_title="Practice Analytics", page_icon="🏀", layout="wide")

OUTCOME_COLORS = alt.Scale(domain=["make", "miss", "unknown"], range=["#2E7D4F", "#C0392B", "#8A94A6"])
KEYFRAME_ORDER = [("set", "Set point"), ("release", "Release"), ("follow_through", "Follow-through")]


@st.cache_resource
def get_store(path: str) -> SessionStore:
    return SessionStore(path)


def shots_frame(shots) -> pd.DataFrame:
    rows = []
    for s in shots:
        row = {"Shot": s["shot_no"], "Time (s)": round(s["t"], 1), "Type": s["kind"].replace("_", " "),
               "Result": s["outcome"], "Cue": s["cue"]}
        row.update({metric_label(k): v for k, v in s["metrics"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def in_range(metric: str, value) -> str:
    if metric not in TARGETS or value is None:
        return ""
    _, _, low, high, *_ = TARGETS[metric]
    return "✅" if low <= value <= high else "⚠️"


def fmt_session(s) -> str:
    when = (s["started"] or "").replace("T", " ")
    return f"{when}  ·  {s['mode']}  ·  {s['id']}"


db_path = st.sidebar.text_input("Database", os.environ.get("COACH_DB", DEFAULT_DB))
if not os.path.exists(db_path):
    st.info("No sessions yet. Record one with `python live.py` or analyse a video with `python analyze.py <video>`.")
    st.stop()
store = get_store(db_path)
sessions = store.sessions()
if not sessions:
    st.info("No sessions yet. Record one with `python live.py` or analyse a video with `python analyze.py <video>`.")
    st.stop()

session = st.sidebar.selectbox("Session", sessions, format_func=fmt_session)
shots = store.shots(session["id"])
dribbles = store.dribbles(session["id"])
df = shots_frame(shots)

st.title("Practice analytics")
st.caption(f"Session {session['id']} · {session['mode']} · shooting hand: {session['shooting_hand']}")

known = [s for s in shots if s["outcome"] in ("make", "miss")]
makes = sum(s["outcome"] == "make" for s in shots)
layups = sum(s["kind"] == "layup" for s in shots)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Shots", len(shots))
c2.metric("Makes", f"{makes}/{len(known)}" if known else "–")
c3.metric("Field goal %", f"{100 * makes / len(known):.0f}%" if known else "–")
c4.metric("Dribbles", len(dribbles))
c5.metric("Duration", f"{(session['duration_s'] or 0) / 60:.1f} min")
if shots and not known:
    st.caption("Make/miss needs the rim marked and the ball detector; this session has neither result.")

tab_overview, tab_shots, tab_trends, tab_dribble, tab_compare, tab_summary = st.tabs(
    ["Overview", "Shots", "Form trends", "Dribbling", "Compare sessions", "Coach summary"])

with tab_overview:
    if df.empty:
        st.write("No shots detected in this session.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Shots over the session")
            chart = alt.Chart(df).mark_circle(size=140).encode(
                x=alt.X("Time (s):Q"), y=alt.Y("Type:N", title=None),
                color=alt.Color("Result:N", scale=OUTCOME_COLORS),
                tooltip=["Shot", "Time (s)", "Type", "Result", "Cue"])
            st.altair_chart(chart, use_container_width=True)
            if len(known) >= 3:
                kdf = pd.DataFrame({"Shot": [s["shot_no"] for s in known],
                                    "make": [s["outcome"] == "make" for s in known]})
                kdf["Rolling FG% (5 shots)"] = kdf["make"].rolling(5, min_periods=1).mean() * 100
                st.subheader("Rolling field goal %")
                st.altair_chart(alt.Chart(kdf).mark_line(point=True).encode(
                    x="Shot:Q", y=alt.Y("Rolling FG% (5 shots):Q", scale=alt.Scale(domain=[0, 100]))),
                    use_container_width=True)
        with right:
            st.subheader("Most common corrections")
            cues = df[~df["Cue"].str.startswith(("Great", "Good"))]["Cue"].value_counts().reset_index()
            cues.columns = ["Cue", "Shots"]
            st.dataframe(cues, hide_index=True, use_container_width=True)
            st.subheader("Shot types")
            st.dataframe(df.groupby("Type").agg(Shots=("Shot", "count"),
                                                Makes=("Result", lambda r: (r == "make").sum())),
                         use_container_width=True)

with tab_shots:
    if df.empty:
        st.write("No shots detected in this session.")
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("Download shot log (CSV)", df.to_csv(index=False).encode(),
                           file_name=f"{session['id']}_shots.csv", mime="text/csv")
        pick = st.selectbox("Inspect shot", [s["shot_no"] for s in shots],
                            format_func=lambda n: f"Shot {n}")
        shot = next(s for s in shots if s["shot_no"] == pick)
        st.markdown(f"**Shot {pick}** · {shot['kind'].replace('_', ' ')} · **{shot['outcome']}** · "
                    f"cue: _{shot['cue']}_")
        cols = st.columns(3)
        for col, (key, label) in zip(cols, KEYFRAME_ORDER):
            rel = shot["keyframes"].get(key)
            path = os.path.join(session["session_dir"] or "", rel) if rel else None
            if path and os.path.exists(path):
                col.image(path, caption=label, use_container_width=True)
            else:
                col.caption(f"{label}: no frame")
        rows = []
        for metric, value in shot["metrics"].items():
            target = ""
            if metric in TARGETS:
                _, unit, low, high, *_ = TARGETS[metric]
                target = f"{low} – {high} {unit}"
            rows.append({"Metric": metric_label(metric), "Value": value, "Target": target,
                         "": in_range(metric, value)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_trends:
    available = sorted({k for s in shots for k, v in s["metrics"].items() if v is not None})
    if not available:
        st.write("No form metrics yet.")
    else:
        metric = st.selectbox("Metric", available, format_func=metric_label,
                              index=available.index("arc_angle") if "arc_angle" in available else 0)
        mdf = pd.DataFrame([{"Shot": s["shot_no"], "Value": s["metrics"].get(metric), "Result": s["outcome"]}
                            for s in shots if s["metrics"].get(metric) is not None])
        layers = []
        if metric in TARGETS:
            _, unit, low, high, *_ = TARGETS[metric]
            band = pd.DataFrame({"low": [low], "high": [min(high, mdf["Value"].max() * 1.2 + 1)]})
            layers.append(alt.Chart(band).mark_rect(opacity=0.12, color="#2E7D4F").encode(y="low:Q", y2="high:Q"))
            st.caption(f"Target {low} – {high} {unit} (shaded)")
        layers.append(alt.Chart(mdf).mark_line(color="#8A94A6").encode(x="Shot:Q", y="Value:Q"))
        layers.append(alt.Chart(mdf).mark_circle(size=90).encode(
            x="Shot:Q", y=alt.Y("Value:Q", title=metric_label(metric)),
            color=alt.Color("Result:N", scale=OUTCOME_COLORS), tooltip=["Shot", "Value", "Result"]))
        st.altair_chart(alt.layer(*layers), use_container_width=True)

        st.subheader("Consistency")
        rows = []
        for m in available:
            vals = [s["metrics"][m] for s in shots if s["metrics"].get(m) is not None]
            row = {"Metric": metric_label(m), "Shots": len(vals), "Mean": round(sum(vals) / len(vals), 2),
                   "Spread (std)": round(pd.Series(vals).std(), 2) if len(vals) > 1 else None, "In target": "–"}
            if m in TARGETS:
                _, _, low, high, *_ = TARGETS[m]
                row["In target"] = f"{100 * sum(low <= v <= high for v in vals) / len(vals):.0f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Lower spread = more repeatable form. Angles are 2D estimates from one camera position.")

with tab_dribble:
    if not dribbles:
        st.write("No dribbles detected (needs the ball detector).")
    else:
        ddf = pd.DataFrame(dribbles)
        span = ddf["t"].max() - ddf["t"].min()
        a, b, c = st.columns(3)
        a.metric("Dribbles", len(ddf))
        b.metric("Pace", f"{(len(ddf) - 1) / span:.1f} /s" if span > 0 else "–")
        heights = ddf["height"].dropna()
        c.metric("Avg height (× body height)", f"{heights.mean():.2f}" if len(heights) else "–")
        st.altair_chart(alt.Chart(ddf.dropna(subset=["height"])).mark_circle().encode(
            x=alt.X("t:Q", title="Time (s)"), y=alt.Y("height:Q", title="Dribble height (body heights)"),
            color="hand:N"), use_container_width=True)
        st.dataframe(ddf["hand"].fillna("unknown").value_counts().rename("Dribbles"), use_container_width=True)

with tab_compare:
    rows = []
    for s in sessions:
        ss = store.shots(s["id"])
        k = [x for x in ss if x["outcome"] in ("make", "miss")]
        row = {"Session": (s["started"] or s["id"])[:16].replace("T", " "), "Shots": len(ss),
               "FG%": round(100 * sum(x["outcome"] == "make" for x in k) / len(k)) if k else None}
        for m in ("arc_angle", "knee_bend", "elbow_set", "follow_through"):
            vals = [x["metrics"].get(m) for x in ss if x["metrics"].get(m) is not None]
            row[metric_label(m)] = round(sum(vals) / len(vals), 1) if vals else None
        rows.append(row)
    cdf = pd.DataFrame(rows).sort_values("Session")
    st.dataframe(cdf, hide_index=True, use_container_width=True)
    if cdf["FG%"].notna().sum() >= 2:
        st.altair_chart(alt.Chart(cdf.dropna(subset=["FG%"])).mark_line(point=True).encode(
            x="Session:N", y=alt.Y("FG%:Q", scale=alt.Scale(domain=[0, 100]))), use_container_width=True)

with tab_summary:
    if session["summary"]:
        st.markdown(session["summary"])
    else:
        st.write("No summary yet for this session.")
    if st.button("Write summary with Claude", disabled=not shots):
        with st.spinner("Writing summary..."):
            payload = [{k: s[k] for k in ("shot_no", "t", "kind", "outcome", "metrics", "cue")} for s in shots]
            text = session_summary({"duration_s": session["duration_s"], "shots": len(shots)}, payload,
                                   {"dribbles": len(dribbles)})
        if text:
            store.set_summary(session["id"], text)
            st.rerun()
        else:
            st.error("Claude is not configured. Set ANTHROPIC_API_KEY (or run `ant auth login`) and try again.")
