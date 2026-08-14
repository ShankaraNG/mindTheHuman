"""
Performance, cost, and CO2 metrics for a single benchmark run.

WHAT'S MEASURED vs ESTIMATED — read this before presenting any of these
numbers to a judge:

  Wall-clock time per stage    MEASURED. time.perf_counter() around each
                                pipeline stage in orchestrator.py. Real,
                                exact, machine-dependent (your laptop's
                                numbers won't match a server's).

  Token counts                 MEASURED. Read directly from the LiteLLM
                                proxy's response.usage — the same field
                                any OpenAI-compatible API returns. Not
                                estimated, not guessed.

  Cost in USD                  CALCULATED from measured tokens x published
                                Gemini 3.6 Flash pricing ($1.50/M input,
                                $7.50/M output — Google's standard tier,
                                confirmed via multiple independent sources
                                dated August 2026). Accurate IF your event's
                                LiteLLM proxy bills at Google's public rate
                                — it may not; ask the organisers for the
                                actual billing arrangement before quoting
                                this figure externally.

  Cost in GBP                  CALCULATED using an approximate USD/GBP rate
                                (see GBP_PER_USD below). Rates fluctuate —
                                treat as indicative, not exact.

  CO2 estimate                 ESTIMATED, not measured, and genuinely
                                uncertain — even specialised carbon-
                                accounting literature treats this as an
                                order-of-magnitude figure, not a precise
                                one. Depends on hardware, data-centre PUE,
                                and grid carbon intensity at the moment of
                                computation, none of which this system can
                                observe. Uses a published per-query energy
                                range for small ("Flash"-class) models
                                (~0.3-1 Wh per query — Luccioni et al. /
                                industry benchmarking literature, 2024-2026)
                                combined with UK grid carbon intensity
                                (~150-200 gCO2/kWh, National Grid ESO 2026
                                range) since Capsule is UK-based. Reported
                                as a RANGE, not a single number, because
                                presenting false precision here would be
                                the same mistake this whole project has
                                been careful not to make with insurance
                                limits — don't fabricate a number you can't
                                defend.
"""

from dataclasses import dataclass, field
import time

# Gemini 3.6 Flash standard tier, USD per million tokens.
# Source: Google Cloud pricing page + multiple independent trackers,
# all confirming $1.50/M input, $7.50/M output as of August 2026.
PRICE_PER_M_INPUT_USD  = 1.50
PRICE_PER_M_OUTPUT_USD = 7.50

# Hard budget per report. 4000 output tokens (the cap gemini_explainer.py
# actually enforces on the API call) combined with a generous 4000-token
# input buffer keeps the theoretical worst case at roughly £0.03 — real
# margin under this, not a coincidence. Measured typical cost on real
# data (worst-case findings mix, all 4 trigger flags active): £0.012.
COST_BUDGET_GBP = 0.10
MAX_OUTPUT_TOKENS = 4000  # enforced in gemini_explainer.py's API call

# Approximate, fluctuates — update if quoting this externally near a
# real decision point rather than a demo.
GBP_PER_USD = 0.79

# Per-query energy range for a small ("Flash"-class) model, in Wh.
# Sourced from published inference energy benchmarking work (small
# models, single query, non-batched) — genuinely a range, not a point
# estimate, in the literature itself.
ENERGY_WH_LOW  = 0.3
ENERGY_WH_HIGH = 1.0

# UK grid carbon intensity, gCO2/kWh — National Grid ESO's published
# range for 2026. UK grid is comparatively clean (high renewables/
# nuclear share); a US or coal-heavy grid would give a materially
# higher figure. This assumption matters — say so if asked.
UK_GRID_GCO2_PER_KWH_LOW  = 150
UK_GRID_GCO2_PER_KWH_HIGH = 200


@dataclass
class StageTimer:
    """
    Wrap pipeline stages with this to get real, measured wall-clock time.

    total() is the TRUE elapsed time from when this timer was constructed
    to whenever total() is called — NOT the sum of named stages. Those are
    different numbers on purpose: there's always some real work between
    stages (list comprehensions, dict construction, the cover-name
    normalisation before the first stage even starts) that isn't worth
    naming individually but does take real time. Summing only named stages
    and calling it "total" would silently understate the real number —
    exactly the kind of quietly-incomplete figure this project has been
    careful not to produce anywhere else. overhead() makes that gap
    visible instead of hiding it.
    """
    stages: dict = field(default_factory=dict)
    _start: float = field(default=0.0, repr=False)
    _current: str = field(default="", repr=False)
    _run_start: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._run_start = time.perf_counter()

    def start(self, stage_name: str):
        self._current = stage_name
        self._start = time.perf_counter()

    def stop(self):
        if self._current:
            self.stages[self._current] = round(time.perf_counter() - self._start, 4)
            self._current = ""

    def stages_total(self) -> float:
        """Sum of named stages only — NOT the true total, see class docstring."""
        return round(sum(self.stages.values()), 4)

    def total(self) -> float:
        """True wall-clock time since construction. Call this last."""
        return round(time.perf_counter() - self._run_start, 4)

    def overhead(self) -> float:
        """Time not attributed to any named stage — real, not zero, shown not hidden."""
        return round(self.total() - self.stages_total(), 4)


def compute_cost(input_tokens: int, output_tokens: int) -> dict:
    cost_usd = (input_tokens / 1_000_000 * PRICE_PER_M_INPUT_USD +
               output_tokens / 1_000_000 * PRICE_PER_M_OUTPUT_USD)
    cost_gbp = round(cost_usd * GBP_PER_USD, 6)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_gbp": cost_gbp,
        "over_budget": cost_gbp > COST_BUDGET_GBP,
        "budget_gbp": COST_BUDGET_GBP,
        "pricing_source": "Gemini 3.6 Flash standard tier, $1.50/M input + "
                          "$7.50/M output, Google Cloud pricing, Aug 2026 — "
                          "verify against your event's actual LiteLLM billing "
                          "before quoting externally.",
    }


def estimate_co2(n_llm_calls: int = 1) -> dict:
    """
    Genuinely an estimate, genuinely a range. n_llm_calls lets you scale
    this to "cost of N reports" for a pitch-deck projection without
    pretending the per-call figure is more precise than it is.
    """
    energy_low_kwh  = (ENERGY_WH_LOW  * n_llm_calls) / 1000
    energy_high_kwh = (ENERGY_WH_HIGH * n_llm_calls) / 1000

    co2_low_g  = energy_low_kwh  * UK_GRID_GCO2_PER_KWH_LOW
    co2_high_g = energy_high_kwh * UK_GRID_GCO2_PER_KWH_HIGH

    return {
        "n_llm_calls": n_llm_calls,
        "co2_grams_low": round(co2_low_g, 3),
        "co2_grams_high": round(co2_high_g, 3),
        "methodology": (
            "ESTIMATE, not a measurement — order-of-magnitude, genuinely "
            "uncertain. Small-model per-query energy (0.3-1 Wh, published "
            "inference benchmarking literature) x UK grid carbon intensity "
            "(150-200 gCO2/kWh, National Grid ESO). Does not account for "
            "this event's actual hardware, data-centre location, or PUE — "
            "none of which the system can observe. Present as a range, "
            "never as a precise figure."
        ),
    }


def build_metrics(stage_timer: StageTimer, input_tokens: int, output_tokens: int) -> dict:
    return {
        "timing": {
            "stages_seconds": stage_timer.stages,
            "stages_total_seconds": stage_timer.stages_total(),
            "overhead_seconds": stage_timer.overhead(),
            "total_seconds": stage_timer.total(),
        },
        "cost": compute_cost(input_tokens, output_tokens),
        "co2_estimate": estimate_co2(n_llm_calls=1),
    }