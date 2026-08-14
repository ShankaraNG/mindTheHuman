"""
Standalone metrics PDF — deliberately separate from pdf_report.py's main
benchmarking report. A broker's cover report and an internal cost/CO2/
performance log serve different audiences and shouldn't be bundled into
one document a client might see.

build_metrics_pdf()   -> bytes, for saving or serving separately
save_metrics_log()    -> writes to data/metrics_logs/, one file per report,
                          timestamped so a run history accumulates rather
                          than overwriting the last one
"""

from datetime import datetime
from pathlib import Path
from fpdf import FPDF

from pdf_report import _sanitize


def _fmt(n, unit=""):
    return f"{n}{unit}" if n is not None else "—"


class MetricsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, "Capsule Cover - Internal Metrics Log", ln=True)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "Not for client distribution - operational log only.", ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)


def build_metrics_pdf(report: dict) -> bytes:
    metrics = report.get("metrics", {})
    profile = report.get("profile", {})
    timing = metrics.get("timing", {})
    cost = metrics.get("cost", {})
    co2 = metrics.get("co2_estimate", {})

    pdf = MetricsPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Run context", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Generated: {datetime.now().isoformat(timespec='seconds')}", ln=True)
    pdf.cell(0, 5, _sanitize(f"Vertical: {profile.get('vertical','-')}  "
                  f"Funding stage: {profile.get('funding_stage','-')}"), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Cost", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Input tokens: {cost.get('input_tokens',0):,}   "
                  f"Output tokens: {cost.get('output_tokens',0):,}   "
                  f"Total: {cost.get('total_tokens',0):,}", ln=True)
    pdf.cell(0, 5, f"Cost: GBP {cost.get('cost_gbp',0):.4f} "
                  f"(USD {cost.get('cost_usd',0):.4f})   "
                  f"Budget: GBP {cost.get('budget_gbp',0):.2f}   "
                  f"Over budget: {cost.get('over_budget', False)}", ln=True)
    pdf.set_font("Helvetica", "I", 7)
    pdf.multi_cell(0, 4, _sanitize(cost.get("pricing_source", "")))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Timing (measured, real wall-clock)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for stage, seconds in timing.get("stages_seconds", {}).items():
        pdf.cell(0, 5, _sanitize(f"  {stage.replace('_',' ')}: {seconds:.4f}s"), ln=True)
    pdf.cell(0, 5, f"  unattributed overhead: {timing.get('overhead_seconds',0):.4f}s", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, f"  TOTAL: {timing.get('total_seconds',0):.4f}s", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "CO2 (estimate, not a measurement)", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Range: {co2.get('co2_grams_low',0):.3f}g - "
                  f"{co2.get('co2_grams_high',0):.3f}g CO2e", ln=True)
    pdf.set_font("Helvetica", "I", 7)
    pdf.multi_cell(0, 4, _sanitize(co2.get("methodology", "")))

    return bytes(pdf.output())


def save_metrics_log(report: dict, out_dir: str = "data/metrics_logs") -> str:
    """Writes to disk, timestamped so a run history accumulates. Returns the path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    vertical = report.get("profile", {}).get("vertical", "unknown")
    fname = out_path / f"metrics_{vertical}_{ts}.pdf"
    fname.write_bytes(build_metrics_pdf(report))
    return str(fname)