"""
PDF export — turns an orchestrator report dict into a downloadable PDF.

Brief requirement: "downloadable output document (PDF or similar)".
Uses fpdf2 — pure Python, no system dependencies (unlike weasyprint), so it
works in any deployment environment without extra install steps.
"""

from fpdf import FPDF

RAG_LABEL = {
    "Immediate action":  "IMMEDIATE ACTION",
    "For consideration": "FOR CONSIDERATION",
    "Review at renewal": "REVIEW AT RENEWAL",
}
DIR_LABEL = {"GAP": "Not held", "INCREASE": "Below benchmark", "OK": "Adequate"}


def _fmt_gbp(n):
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"GBP {n/1_000_000:.1f}m"
    if n >= 1_000:
        return f"GBP {n/1_000:.0f}k"
    return f"GBP {n:,.0f}"


# fpdf2's core Helvetica font is latin-1 only. Real text — Gemini output,
# broker recommendation text from deck_recommendations.csv — is full of
# unicode punctuation (em-dash, curly quotes, ellipsis) that crashes PDF
# generation. Sanitize rather than bundle a Unicode TTF font file, which
# would add a deployment dependency.
_UNICODE_REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    "\u2022": "-", "\u00b7": "-",
}


def _sanitize(text) -> str:
    if text is None:
        return ""
    s = str(text)
    for uni, ascii_eq in _UNICODE_REPLACEMENTS.items():
        s = s.replace(uni, ascii_eq)
    return s.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Capsule Cover - Benchmarking Report", ln=True)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "Guidance, not regulated advice.", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(235, 235, 235)
        self.cell(0, 8, text, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, _sanitize(text))
        self.ln(1)

    def priority_badge(self, priority):
        colors = {
            "Immediate action":  (231, 76, 60),
            "For consideration": (230, 165, 25),
            "Review at renewal": (39, 141, 82),
        }
        r, g, b = colors.get(priority, (100, 100, 100))
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        label = RAG_LABEL.get(priority, priority)
        w = self.get_string_width(label) + 6
        self.cell(w, 6, label, fill=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(8)


def build_pdf(report: dict) -> bytes:
    profile = report.get("profile", {})
    stats   = report.get("summary_stats", {})
    gaps    = report.get("gap_results", [])
    exp     = report.get("explanation", {})

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Profile summary ─────────────────────────────────────────────────
    pdf.section_title("Business profile")
    pdf.body_text(
        f"Vertical: {profile.get('vertical', '-')}    "
        f"Funding stage: {profile.get('funding_stage') or 'unknown'}\n"
        f"Turnover: {_fmt_gbp(profile.get('turnover_gbp'))}    "
        f"Total funding raised: {_fmt_gbp(profile.get('total_funding_gbp'))}\n"
        f"Employees: {profile.get('employee_count', '-')}    "
        f"US revenue: {profile.get('us_revenue_pct') or 0}%"
    )
    pdf.ln(2)

    pdf.body_text(
        f"Benchmarked against {stats.get('cohort_size', 0)} comparable "
        f"{profile.get('vertical','')} companies "
        f"({stats.get('n_with_cover_data', 0)} with real cover data, "
        f"{stats.get('cover_data_coverage_pct', 0)}% coverage)."
    )
    pdf.ln(3)

    # ── Executive summary ────────────────────────────────────────────────
    if exp.get("executive_summary"):
        pdf.section_title("Summary")
        pdf.body_text(exp["executive_summary"])
        pdf.ln(2)

    # ── Findings, grouped by priority ────────────────────────────────────
    exp_map = {f["cover"]: f for f in exp.get("cover_findings", [])}
    order = {"Immediate action": 0, "For consideration": 1, "Review at renewal": 2}
    sorted_gaps = sorted(gaps, key=lambda g: order.get(g["priority"], 3))

    if sorted_gaps:
        pdf.section_title("Findings")

    for item in sorted_gaps:
        cover     = item["cover"]
        direction = DIR_LABEL.get(item["direction"], item["direction"])
        ev        = item.get("evidence", {})
        rationale = exp_map.get(cover, {}).get("rationale", "")

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, _sanitize(f"{cover} - {direction}"), ln=True)
        pdf.priority_badge(item["priority"])

        detail_bits = []
        if item.get("current_limit"):
            detail_bits.append(f"Current: {_fmt_gbp(item['current_limit'])}")
        else:
            detail_bits.append("Current: Not held")
        if ev.get("peer_median"):
            detail_bits.append(f"Peer median ({ev.get('peer_n')} peers): {_fmt_gbp(ev['peer_median'])}")
        if ev.get("band_rec"):
            detail_bits.append(f"Capsule band: {_fmt_gbp(ev['band_rec'])}")
        pdf.body_text("   ".join(detail_bits), size=9)

        if rationale:
            pdf.body_text(rationale, size=9)

        alternatives = exp_map.get(cover, {}).get("alternatives", [])
        if alternatives:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Alternative approaches:", ln=True)
            for alt in alternatives:
                marker = "[grounded] " if alt.get("grounded") else "[option] "
                pdf.body_text(f"- {marker}{alt.get('approach','')} - {alt.get('detail','')}", size=9)

        pdf.ln(2)

    # ── Priority actions ──────────────────────────────────────────────────
    if exp.get("top_actions"):
        pdf.section_title("Priority actions")
        for a in exp["top_actions"]:
            pdf.body_text(f"[{RAG_LABEL.get(a['priority'], a['priority'])}]  {a['action']}", size=9)
        pdf.ln(2)

    # ── Disclaimer ────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 4.5, exp.get("disclaimer") or
                   "This is guidance, not regulated advice. Speak with a Capsule "
                   "broker to confirm suitability for your circumstances.")

    return bytes(pdf.output())