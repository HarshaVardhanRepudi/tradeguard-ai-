"""
TradeGuard AI — Salary Matching Rules Engine
----------------------------------------------
Pure, deterministic logic. No database or LLM calls. This is the function
the agent calls as a "tool" to check whether a proposed trade satisfies
NBA CBA salary-matching rules, based on the outgoing team's apron status.

Rules encoded here (2025-26 season figures):
  - Under first apron:      incoming salary <= 125% of outgoing + $100,000
  - Over first apron only:  incoming salary <= 110% of outgoing + $100,000
  - Over second apron:      no aggregation — if outgoing side has more than
                             one contract, the trade is not permitted at all,
                             regardless of dollar amounts. Single-for-single
                             (or sending out more than you receive) is still fine.

This is intentionally a simplified model of the real CBA (which has more
edge cases: trade exceptions, base-year compensation, minimum-salary
exceptions, etc.) — documented as a scoping decision, not an oversight.
"""

from dataclasses import dataclass, field
from enum import Enum


class ApronStatus(str, Enum):
    UNDER_CAP = "under_cap"
    OVER_CAP = "over_cap"
    OVER_TAX = "over_tax"
    FIRST_APRON = "first_apron"
    SECOND_APRON = "second_apron"


@dataclass
class TradeProposal:
    team_from: str
    team_to: str
    outgoing_salaries: list  # list of floats, one per player sent out
    incoming_salaries: list  # list of floats, one per player received
    team_from_apron_status: ApronStatus


@dataclass
class MatchResult:
    permitted: bool
    reason: str
    outgoing_total: float
    incoming_total: float
    max_incoming_allowed: float | None
    rule_applied: str
    citation: str


def check_salary_match(trade: TradeProposal) -> MatchResult:
    outgoing_total = sum(trade.outgoing_salaries)
    incoming_total = sum(trade.incoming_salaries)
    status = trade.team_from_apron_status

    # Second apron: aggregation ban. Multiple outgoing contracts -> not allowed,
    # full stop, regardless of dollar math.
    if status == ApronStatus.SECOND_APRON and len(trade.outgoing_salaries) > 1:
        return MatchResult(
            permitted=False,
            reason=(
                f"{trade.team_from} is over the second apron and is trying to send out "
                f"{len(trade.outgoing_salaries)} contracts in one trade. Teams over the "
                f"second apron cannot aggregate multiple outgoing contracts to match a "
                f"larger incoming salary — they are limited to one-for-one trades or "
                f"sending out more salary than they receive."
            ),
            outgoing_total=outgoing_total,
            incoming_total=incoming_total,
            max_incoming_allowed=None,
            rule_applied="second_apron_aggregation_ban",
            citation="CBA Article VII, Sec. 5(a) — Second Apron Aggregation Restriction",
        )

    # Determine the matching percentage based on apron status.
    if status == ApronStatus.SECOND_APRON:
        # single-for-single or sending out more than receiving is allowed,
        # but still capped at 100% (no upside) once over the second apron.
        pct, rule_name, citation = 1.00, "second_apron_single_for_single", \
            "CBA Article VII, Sec. 5(b) — Second Apron Salary Matching"
    elif status == ApronStatus.FIRST_APRON:
        pct, rule_name, citation = 1.10, "first_apron_matching", \
            "CBA Article VII, Sec. 3(c) — First Apron Salary Matching (110%)"
    else:
        pct, rule_name, citation = 1.25, "standard_matching", \
            "CBA Article VII, Sec. 3(a) — Standard Salary Matching (125%)"

    max_incoming_allowed = outgoing_total * pct + 100_000
    permitted = incoming_total <= max_incoming_allowed

    reason = (
        f"{trade.team_from} sends out ${outgoing_total:,.0f} and would receive "
        f"${incoming_total:,.0f}. Under the '{rule_name}' rule, they may take back up to "
        f"${max_incoming_allowed:,.0f} ({pct*100:.0f}% + $100,000). "
        f"{'This is within the allowed limit.' if permitted else 'This exceeds the allowed limit.'}"
    )

    return MatchResult(
        permitted=permitted,
        reason=reason,
        outgoing_total=outgoing_total,
        incoming_total=incoming_total,
        max_incoming_allowed=max_incoming_allowed,
        rule_applied=rule_name,
        citation=citation,
    )


if __name__ == "__main__":
    # Worked example matching the one in the design doc
    trade = TradeProposal(
        team_from="Team A",
        team_to="Team B",
        outgoing_salaries=[15_000_000],
        incoming_salaries=[18_000_000],
        team_from_apron_status=ApronStatus.FIRST_APRON,
    )
    result = check_salary_match(trade)
    print(f"Permitted: {result.permitted}")
    print(f"Reason: {result.reason}")
    print(f"Rule: {result.rule_applied}  |  Citation: {result.citation}")
