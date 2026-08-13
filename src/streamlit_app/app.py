"""
Capsule Cover — Benchmarking Engine (Streamlit)

Run: streamlit run src/streamlit_app/app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st

_SRC = Path(__file__).parent.parent
for _sub in ("", "profile", "benchmark", "engine"):
    _p = str(_SRC / _sub) if _sub else str(_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.questionnaire_parser import parse_questionnaire_file
from engine.orchestrator import Orchestrator
from engine.pdf_report import build_pdf
from profile.normalise import normalise_intake

REPO_ROOT       = Path(__file__).parent.parent.parent
DATA_DIR        = REPO_ROOT / "data"
HEALTHCHECK_DIR = REPO_ROOT / "dataset" / "1_recommendation_bank"
if not HEALTHCHECK_DIR.exists():
    HEALTHCHECK_DIR = None

VERTICALS = ["Tech", "FinTech", "Consumer", "MedTech"]
FUNDING_STAGES = ["Bootstrapped", "Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Other"]

COVER_OPTIONS = {
    "Tech":     ["Professional Indemnity", "Cyber", "Employers Liability", "Directors & Officers",
                 "Crime", "Employment Practices Liability", "Business Interruption"],
    "FinTech":  ["Professional Indemnity", "Cyber", "Crime", "Employers Liability",
                 "Directors & Officers", "Employment Practices Liability", "Business Interruption"],
    "Consumer": ["Public & Products Liability", "Employers Liability", "Product Recall",
                 "Directors & Officers", "Cyber", "Business Interruption", "Goods in Transit"],
    "MedTech":  ["Professional Indemnity", "Public & Products Liability", "Employers Liability",
                 "Directors & Officers", "Cyber", "Product Recall", "Clinical Trials",
                 "Medical Malpractice", "Business Interruption"],
}

st.set_page_config(page_title="Capsule Cover — Benchmarking", page_icon="🛡️", layout="centered")


@st.cache_resource
def get_orchestrator():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    return Orchestrator(
        data_dir=str(DATA_DIR),
        healthcheck_dir=str(HEALTHCHECK_DIR) if HEALTHCHECK_DIR else None,
        gemini_api_key=api_key,
    )


def fmt_gbp(n):
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"£{n/1_000_000:.1f}m"
    if n >= 1_000:
        return f"£{n/1_000:.0f}k"
    return f"£{n:,.0f}"


RAG_COLORS = {"Immediate action": "🔴", "For consideration": "🟠", "Review at renewal": "🟢"}
DIR_LABELS = {"GAP": "⬛ Not held", "INCREASE": "⬆️ Below benchmark",
              "OK": "✅ Adequate", "NEEDS_REVIEW": "❔ Insufficient evidence"}

for k in ["step", "profile", "report", "current_cover"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.step is None:
    st.session_state.step = "upload"

st.markdown("## 🛡️ Capsule Cover — Cover benchmarking engine")
st.caption("Guidance, not regulated advice. Benchmarked against Capsule's own client book.")
st.divider()


# ══════════════════════════ STEP 1 — Upload ═══════════════════════════════
if st.session_state.step == "upload":
    st.markdown("### Step 1 — Upload the health check questionnaire")
    uploaded = st.file_uploader("Questionnaire file", type=["csv", "xlsx", "xls"])

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Skip upload, enter manually", use_container_width=True):
            st.session_state.profile = normalise_intake({
                "vertical": "Tech", "funding_series": "Series A",
                "customer_type": "B2B", "software_or_hardware": "Software",
            })
            st.session_state.step = "profile"
            st.rerun()

    if uploaded is not None:
        try:
            parsed = parse_questionnaire_file(uploaded)
            profiles = parsed["profiles"]

            if len(profiles) > 1:
                st.info(f"File has {len(profiles)} rows. Select which client to benchmark:")
                labels = [f"{p['_client_id']} — {p['vertical']} — {fmt_gbp(p['turnover_gbp'])} turnover"
                          for p in profiles]
                choice = st.selectbox("Select client", range(len(profiles)),
                                       format_func=lambda i: labels[i])
                selected = profiles[choice]
            else:
                selected = profiles[0]

            st.success(f"Parsed profile for {selected.get('_client_id', 'the uploaded client')}")
            with st.expander("Preview extracted profile"):
                st.json({k: v for k, v in selected.items() if not k.startswith(("_",))})

            if st.button("Continue with this profile →", type="primary"):
                st.session_state.profile = selected
                st.session_state.step = "profile"
                st.rerun()

        except Exception as e:
            st.error(f"Couldn't parse file: {e}")


# ══════════════════════════ STEP 2 — Confirm profile ═══════════════════════
elif st.session_state.step == "profile":
    st.markdown("### Step 2 — Confirm the business profile")
    p = st.session_state.profile

    col1, col2 = st.columns(2)
    with col1:
        vertical = st.selectbox("Vertical", VERTICALS,
                                 index=VERTICALS.index(p["vertical"]) if p.get("vertical") in VERTICALS else 0)
        funding = st.selectbox("Funding stage (self-reported)", FUNDING_STAGES, index=2,
                                help="Used with your CRM stage to place you on Capsule's maturity ladder.")
        turnover = st.number_input("Annual turnover (£)", min_value=0,
                                    value=int(p.get("turnover_gbp") or 0), step=100000)
        total_funding = st.number_input("Total funding raised (£)", min_value=0,
                                         value=int(p.get("total_funding_gbp") or 0), step=100000)
    with col2:
        employees = st.number_input("Employees", min_value=0, value=int(p.get("employee_count") or 0))
        us_pct = st.number_input("US revenue (%)", min_value=0, max_value=100,
                                  value=int(p.get("us_revenue_pct") or 0))
        customer_type = st.selectbox("Customer type", ["B2B", "B2C", "Both"],
                                      index=["B2B","B2C","Both"].index(p["customer_type"]) if p.get("customer_type") in ["B2B","B2C","Both"] else 0)
        sw_hw = st.selectbox("Software or hardware", ["Software", "Hardware", "Both"],
                              index=["Software","Hardware","Both"].index(p["software_or_hardware"]) if p.get("software_or_hardware") in ["Software","Hardware","Both"] else 0)

    regulated_display = {True: "Yes", False: "No", None: "Unknown"}[p.get("regulated")]
    regulated_choice = st.radio("FCA regulated?", ["Unknown", "Yes", "No"], horizontal=True,
                                 index=["Unknown","Yes","No"].index(regulated_display))

    business_desc = st.text_area("Business description", value=p.get("business_description") or "",
                                  help="Used to detect personal/medical/financial data and stock/logistics exposure.")

    if st.button("Continue to current cover →", type="primary"):
        st.session_state.profile = normalise_intake({
            "vertical": vertical, "funding_series": funding,
            "turnover_gbp": turnover or None, "total_funding_raised_gbp": total_funding or None,
            "employee_count": employees or None, "us_revenue_pct": float(us_pct),
            "fca_regulated": {"Yes": "true", "No": "false", "Unknown": None}[regulated_choice],
            "customer_type": customer_type, "software_or_hardware": sw_hw,
            "business_description": business_desc or None,
        })
        st.session_state.step = "cover"
        st.rerun()

    if st.button("← Back"):
        st.session_state.step = "upload"
        st.rerun()


# ══════════════════════════ STEP 3 — Current cover ═════════════════════════
elif st.session_state.step == "cover":
    st.markdown("### Step 3 — Current cover")
    st.caption("The questionnaire doesn't capture limits. Tell us what's currently held.")

    vertical = st.session_state.profile["vertical"]
    options = COVER_OPTIONS.get(vertical, COVER_OPTIONS["Tech"])

    cover_selections = {}
    for name in options:
        c1, c2 = st.columns([2, 1])
        with c1:
            held = st.checkbox(name, key=f"chk_{name}")
        with c2:
            if held:
                limit = st.number_input("£m", min_value=0.0, step=0.5,
                                         key=f"limit_{name}", label_visibility="collapsed")
                cover_selections[name] = limit * 1_000_000 if limit > 0 else None

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = "profile"
            st.rerun()
    with col2:
        if st.button("Generate benchmark report →", type="primary", use_container_width=True):
            st.session_state.current_cover = cover_selections
            st.session_state.report = None
            st.session_state.step = "report"
            st.rerun()


# ══════════════════════════ STEP 4 — Report ════════════════════════════════
elif st.session_state.step == "report":
    if st.session_state.report is None:
        with st.spinner("Running pipeline — matching peers, benchmarking cover, detecting gaps…"):
            try:
                orch = get_orchestrator()
                result = orch.run(st.session_state.profile, st.session_state.current_cover)
                st.session_state.report = result
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    report  = st.session_state.report
    profile = report["profile"]
    gaps    = report["gap_results"]
    stats   = report["summary_stats"]
    exp     = report.get("explanation", {})

    st.markdown(f"### Cover benchmarking report — {profile['vertical']} · {profile.get('funding_stage') or 'stage unknown'}")

    st.info(f"Benchmarked against **{stats.get('cohort_size', 0)} comparable {profile['vertical']} companies** "
            f"with real cover-line evidence · cohort {stats.get('cohort_confidence', 'unknown')}")

    if exp.get("executive_summary"):
        st.markdown("#### Summary")
        st.write(exp["executive_summary"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Immediate actions", stats.get("immediate_count", 0))
    c2.metric("For consideration", stats.get("consideration_count", 0))
    c3.metric("Covers assessed", stats.get("covers_assessed", 0))

    exp_map = {f["cover"]: f for f in exp.get("cover_findings", [])}

    st.divider()
    st.markdown("#### Findings")

    for priority_group in ["Immediate action", "For consideration", "Review at renewal"]:
        items = [g for g in gaps if g["priority"] == priority_group]
        if not items:
            continue
        st.markdown(f"**{RAG_COLORS[priority_group]} {priority_group}**")
        for item in items:
            ev = item["evidence"]
            e = exp_map.get(item["cover"], {})
            with st.expander(f"{item['cover']} — {DIR_LABELS.get(item['direction'], item['direction'])} "
                             f"({item['confidence']})"):
                cols = st.columns(3)
                cols[0].markdown(f"**Current**  \n{fmt_gbp(item['current_limit']) if item['current_limit'] else 'Not held'}")
                if ev.get("peer_median"):
                    cols[1].markdown(f"**Peer median** ({ev.get('peer_n')} of {ev.get('n_cohort')} peers)  \n{fmt_gbp(ev['peer_median'])}")
                if ev.get("prevalence") is not None:
                    cols[2].markdown(f"**Prevalence**  \n{ev['prevalence']:.0%}")
                if e.get("rationale"):
                    st.write(e["rationale"])
                if ev.get("note"):
                    st.caption(ev["note"])

    if exp.get("top_actions"):
        st.divider()
        st.markdown("#### Priority actions")
        for a in exp["top_actions"]:
            st.write(f"{RAG_COLORS.get(a['priority'],'•')} {a['action']}")

    st.divider()
    st.caption(f"**Guidance, not regulated advice.** {exp.get('disclaimer', '')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("↺ New assessment"):
            for k in ["step", "profile", "report", "current_cover"]:
                st.session_state[k] = None
            st.session_state.step = "upload"
            st.rerun()
    with col2:
        pdf_bytes = build_pdf(report)
        st.download_button("⬇ Download PDF report", data=pdf_bytes,
                           file_name=f"capsule_benchmark_{profile['vertical']}.pdf",
                           mime="application/pdf", use_container_width=True)