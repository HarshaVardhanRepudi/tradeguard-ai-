"""
Tests for etl/salary_matching.py — the deterministic CBA rules engine.

Run:  python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from etl.salary_matching import TradeProposal, ApronStatus, check_salary_match


def make_trade(status, outgoing, incoming):
    return TradeProposal(
        team_from="TEST", team_to="TEST2",
        outgoing_salaries=outgoing, incoming_salaries=incoming,
        team_from_apron_status=status,
    )


class TestStandardMatching:
    """Under-cap / over-cap / over-tax teams: 125% + $100,000 rule."""

    def test_within_limit_is_permitted(self):
        trade = make_trade(ApronStatus.OVER_CAP, [10_000_000], [12_000_000])
        result = check_salary_match(trade)
        assert result.permitted is True
        assert result.rule_applied == "standard_matching"

    def test_exactly_at_limit_is_permitted(self):
        # 10M * 1.25 + 100,000 = 12,600,000 exactly
        trade = make_trade(ApronStatus.OVER_CAP, [10_000_000], [12_600_000])
        result = check_salary_match(trade)
        assert result.permitted is True

    def test_one_dollar_over_limit_is_rejected(self):
        trade = make_trade(ApronStatus.OVER_CAP, [10_000_000], [12_600_001])
        result = check_salary_match(trade)
        assert result.permitted is False

    def test_under_cap_uses_standard_rule_too(self):
        trade = make_trade(ApronStatus.UNDER_CAP, [5_000_000], [6_000_000])
        result = check_salary_match(trade)
        assert result.permitted is True
        assert result.rule_applied == "standard_matching"


class TestFirstApronMatching:
    """First-apron teams: stricter 110% + $100,000 rule."""

    def test_within_110_percent_is_permitted(self):
        # 15M * 1.10 + 100,000 = 16,600,000
        trade = make_trade(ApronStatus.FIRST_APRON, [15_000_000], [16_600_000])
        result = check_salary_match(trade)
        assert result.permitted is True
        assert result.rule_applied == "first_apron_matching"

    def test_over_110_percent_is_rejected(self):
        trade = make_trade(ApronStatus.FIRST_APRON, [15_000_000], [18_000_000])
        result = check_salary_match(trade)
        assert result.permitted is False

    def test_amount_permitted_under_standard_but_not_first_apron(self):
        """The same trade should flip legality depending on apron tier —
        this is the core behavior the whole project depends on."""
        outgoing, incoming = [15_000_000], [18_000_000]
        standard = check_salary_match(make_trade(ApronStatus.OVER_CAP, outgoing, incoming))
        first_apron = check_salary_match(make_trade(ApronStatus.FIRST_APRON, outgoing, incoming))
        assert standard.permitted is True
        assert first_apron.permitted is False


class TestSecondApronAggregationBan:
    """Second-apron teams: cannot combine multiple outgoing contracts."""

    def test_single_contract_uses_100_percent_rule(self):
        trade = make_trade(ApronStatus.SECOND_APRON, [10_000_000], [10_100_000])
        result = check_salary_match(trade)
        assert result.permitted is True
        assert result.rule_applied == "second_apron_single_for_single"

    def test_single_contract_over_100_percent_is_rejected(self):
        trade = make_trade(ApronStatus.SECOND_APRON, [10_000_000], [10_200_001])
        result = check_salary_match(trade)
        assert result.permitted is False

    def test_multiple_outgoing_contracts_always_rejected(self):
        """The aggregation ban is a flat prohibition — it fires regardless
        of whether the dollar math would otherwise work out."""
        trade = make_trade(ApronStatus.SECOND_APRON, [5_000_000, 5_000_000], [9_000_000])
        result = check_salary_match(trade)
        assert result.permitted is False
        assert result.rule_applied == "second_apron_aggregation_ban"

    def test_aggregation_ban_fires_even_when_dollar_amounts_would_match(self):
        """Explicitly proves the ban is about contract COUNT, not just money —
        this is the trickiest rule in the whole engine and the easiest to
        get subtly wrong."""
        trade = make_trade(ApronStatus.SECOND_APRON, [4_000_000, 4_000_000], [7_500_000])
        result = check_salary_match(trade)
        assert result.permitted is False
        assert "aggregat" in result.reason.lower()


class TestRealWorldCases:
    """Cases matching real teams/scenarios verified earlier in the project."""

    def test_cleveland_second_apron_two_contracts_rejected(self):
        """Real Cleveland Cavaliers 2025-26 scenario (real payroll: $211.98M,
        confirmed second apron)."""
        trade = make_trade(ApronStatus.SECOND_APRON, [6_623_456, 3_492_480], [9_500_000])
        result = check_salary_match(trade)
        assert result.permitted is False

    def test_golden_state_first_apron_strict_matching(self):
        """Real Golden State Warriors 2025-26 scenario (real payroll: $204.85M,
        confirmed first apron)."""
        trade = make_trade(ApronStatus.FIRST_APRON, [5_685_000], [7_500_000])
        result = check_salary_match(trade)
        assert result.permitted is False  # exceeds 110% + $100k

    def test_boston_over_cap_standard_matching(self):
        """Real Boston Celtics 2025-26 scenario (real payroll: $187.42M,
        confirmed over_cap, not apron-restricted)."""
        trade = make_trade(ApronStatus.OVER_CAP, [10_000_000], [11_500_000])
        result = check_salary_match(trade)
        assert result.permitted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
