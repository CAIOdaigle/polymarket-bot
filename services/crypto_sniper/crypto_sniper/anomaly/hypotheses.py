"""Hypothesis ranker — given the set of firing detectors, score the most
likely explanations and recommend a concrete fix for each.

Scoring (deliberately simple — easy to read, easy to extend):

  score = match_quality * prior * 100

  match_quality = (firing detectors matching this hypothesis) /
                  (total detectors that should fire if hypothesis is true)
                  — penalised slightly if other unrelated detectors also fired

  prior = baseline plausibility of the hypothesis (0..1)

We deliberately do NOT use anything more sophisticated (Bayesian net, ML
classifier, etc.) because we want the reasoning to be auditable: the user
should be able to look at a 87/100 score and trace exactly which detectors
contributed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crypto_sniper.anomaly.detectors import DetectorResult


@dataclass
class Hypothesis:
    """Definition of a known failure mode."""
    name: str
    description: str
    expected_detectors: list[str]    # detectors that fire if this is true
    contradicts: list[str] = field(default_factory=list)  # detectors absence supports it
    prior: float = 0.5               # baseline plausibility 0..1
    recommended_fix: str = ""
    fix_confidence: int = 50         # 0-100, how likely the fix resolves it


@dataclass
class RankedHypothesis:
    name: str
    description: str
    score: int                       # 0-100
    matching: list[str]              # detector names that support it
    missing: list[str]               # expected detectors that did NOT fire
    recommended_fix: str
    fix_confidence: int


# ────────────────────────────────────────────────────────────────────────────
# Hypothesis catalog
#
# Add new entries here as failure modes are identified. Each one needs:
#  - which detectors fire when this is true
#  - a prior (how often this is the explanation, before evidence)
#  - the concrete fix
# ────────────────────────────────────────────────────────────────────────────

CATALOG: list[Hypothesis] = [
    Hypothesis(
        name="phantom_clob_liquidity",
        description=(
            "The CLOB /book endpoint shows large nominal depth at a few "
            "round price levels (e.g. $0.30/$0.50/$0.70), but that liquidity "
            "isn't actually fillable in live trading. Dry-run treats it as "
            "real, producing impossible win rates and exponential PnL."
        ),
        expected_detectors=[
            "price_quantization",
            "win_rate_impossible",
            "wins_on_noise",
        ],
        prior=0.85,  # very plausible given Polymarket's known market-maker setup
        recommended_fix=(
            "Submit ONE live $1 FOK order at quote+$0.02 limit. Compare actual "
            "fill price (or kill) to the quoted ask. If fill diverges by >5c "
            "or kills, the depth is phantom — keep dry-run sizing and look "
            "for a different exchange or strategy."
        ),
        fix_confidence=85,
    ),
    Hypothesis(
        name="compound_bet_inflation",
        description=(
            "Strategy may have a small real edge, but Kelly fraction sizing "
            "compounds it on a growing bankroll, ballooning bet sizes far "
            "beyond what the depth/edge can support. The PnL number is real "
            "in dry-run but reflects an unrealistic deployment of capital."
        ),
        expected_detectors=[
            "bankroll_growth",
            "outlier_concentration",
        ],
        prior=0.70,
        recommended_fix=(
            "Add an absolute USD cap on bets in KellyConfig (e.g. $25), applied "
            "after the bankroll-fraction sizing. Reset bankroll to a fixed "
            "starting value and verify trade-by-trade PnL stays in a sane range."
        ),
        fix_confidence=95,  # we already shipped this
    ),
    Hypothesis(
        name="outcome_resolution_leak",
        description=(
            "The window_open or window_close price used to determine WIN/LOSS "
            "is computed from data that the signal already saw, creating a "
            "tautology where signal direction trivially matches 'actual'. "
            "Wins concentrate on tiny moves because the signal IS the outcome."
        ),
        expected_detectors=[
            "wins_on_noise",
            "win_rate_impossible",
        ],
        contradicts=["price_quantization"],  # if data is noisy across prices, prob not this
        prior=0.30,
        recommended_fix=(
            "Audit the timestamp ordering in runner._run_window and "
            "_check_resolution. Confirm window_open is captured BEFORE the "
            "strategy evaluates and btc_close is captured AFTER window-close "
            "from a candle that started at-or-after window-close timestamp."
        ),
        fix_confidence=70,
    ),
    Hypothesis(
        name="calibration_drift",
        description=(
            "TA confidence score is no longer tracking empirical win rate. "
            "Kelly is sizing as if conf=0.9 means 90% probability when the "
            "real frequency is something else, leading to systematic over- "
            "or under-sizing."
        ),
        expected_detectors=["calibration_gap"],
        prior=0.40,
        recommended_fix=(
            "Force a calibration rebuild via build_calibration_from_db(). "
            "If gap persists across multiple windows, consider adding a "
            "confidence cap (e.g. min(stated, empirical_for_bucket))."
        ),
        fix_confidence=75,
    ),
    Hypothesis(
        name="real_edge_real_variance",
        description=(
            "The strategy genuinely has a small edge and we're seeing a hot "
            "streak. Outlier concentration is a property of any heavy-tailed "
            "binary-payout strategy. No structural problem — just need more "
            "samples before drawing conclusions."
        ),
        expected_detectors=["outlier_concentration"],
        contradicts=["price_quantization", "win_rate_impossible", "wins_on_noise"],
        prior=0.20,  # be skeptical: most "edges" aren't real
        recommended_fix=(
            "Continue dry-run for at least 200 more trades and watch for the "
            "win rate to regress toward 55-60%. Add depth verification probes "
            "before considering live capital."
        ),
        fix_confidence=40,  # 'do nothing' isn't really a fix
    ),
]


def rank_hypotheses(
    fired: list[DetectorResult],
    top_k: int = 3,
) -> list[RankedHypothesis]:
    """Score every hypothesis given the set of firing detectors and return
    the top K by score.

    Score interpretation:
      80-100  high confidence — act on the recommended fix
      50-79   plausible — investigate, may need more data
      0-49    weak match — included only because the user asked for top K
    """
    fired_names = {r.name for r in fired}
    if not fired:
        return []

    scored: list[RankedHypothesis] = []
    for h in CATALOG:
        expected = set(h.expected_detectors)
        if not expected:
            continue
        matching = expected & fired_names
        if not matching:
            continue

        # Base match: fraction of expected detectors that fired
        match_quality = len(matching) / len(expected)

        # Penalize if hypotheses' contradictors also fired (weakens confidence)
        contradiction_penalty = sum(1 for c in h.contradicts if c in fired_names)
        if contradiction_penalty:
            match_quality *= max(0.0, 1 - 0.4 * contradiction_penalty)

        # Penalize if many OTHER detectors fired that this hypothesis doesn't predict
        unexplained = (fired_names - expected) - set(h.contradicts)
        if unexplained:
            match_quality *= max(0.5, 1 - 0.1 * len(unexplained))

        score = round(match_quality * h.prior * 100)
        scored.append(RankedHypothesis(
            name=h.name,
            description=h.description,
            score=score,
            matching=sorted(matching),
            missing=sorted(expected - matching),
            recommended_fix=h.recommended_fix,
            fix_confidence=h.fix_confidence,
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
