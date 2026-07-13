"""
Tests for MetricExporter date range calculation in query_aws_cost_explorer().

Regression suite for Issue #42: on the first day of a month, MONTHLY queries
sent Start == End to Cost Explorer, which rejected them with:
    "Start date (and hour) should be before end date (and hour)"
"""

from datetime import datetime as real_datetime
from unittest.mock import MagicMock, patch

import pytest

from app.exporter import MetricExporter


_GROUP_BY_DISABLED = {
    "enabled": False,
    "groups": [],
    "merge_minor_cost": {"enabled": False},
}
_TARGETS = [{"Publisher": "123456789012"}]


@pytest.fixture(autouse=True)
def _mock_gauge(monkeypatch):
    """Stub out prometheus_client.Gauge to prevent registry collisions between tests."""
    monkeypatch.setattr("app.exporter.Gauge", MagicMock)


def _make_exporter(granularity: str = "MONTHLY", data_delay_days: int = 0) -> MetricExporter:
    return MetricExporter(
        polling_interval_seconds=3600,
        metric_name="test_cost_metric",
        aws_access_key="",
        aws_access_secret="",
        aws_assumed_role_name="",
        group_by=_GROUP_BY_DISABLED,
        targets=_TARGETS,
        metric_type="UnblendedCost",
        granularity=granularity,
        data_delay_days=data_delay_days,
    )


def _capture_time_period(exporter: MetricExporter, today: real_datetime) -> dict:
    """
    Invoke query_aws_cost_explorer with a fixed datetime.today() and return the
    TimePeriod dict that was passed to GetCostAndUsage.
    """
    mock_client = MagicMock()
    mock_client.get_cost_and_usage.return_value = {"ResultsByTime": []}

    with patch("app.exporter.datetime") as mock_dt:
        mock_dt.today.return_value = today
        # Forward constructor calls to the real class so datetime(y, m, d) still works.
        mock_dt.side_effect = lambda *args, **kwargs: real_datetime(*args, **kwargs)
        exporter.query_aws_cost_explorer(mock_client, _GROUP_BY_DISABLED)

    return mock_client.get_cost_and_usage.call_args.kwargs["TimePeriod"]


class TestMonthlyGranularity:
    """MONTHLY granularity: verify Start/End dates are valid for Cost Explorer."""

    def test_first_of_month_no_delay_start_before_end(self):
        """
        Regression for Issue #42.

        Before the fix, on the 1st of any month start_date and end_date both
        formatted to the same calendar date, so AWS CE rejected the query with
        "Start date (and hour) should be before end date (and hour)".
        """
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 1, 10, 30, 0))

        assert period["Start"] == "2025-11-01"
        assert period["End"] == "2025-11-02"
        assert period["Start"] < period["End"], "AWS CE requires Start strictly before End"

    def test_mid_month_no_delay(self):
        """Mid-month MTD range: first of month through today-inclusive (End is exclusive in CE)."""
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 15, 14, 0, 0))

        assert period["Start"] == "2025-11-01"
        assert period["End"] == "2025-11-16"

    def test_last_day_of_month_no_delay(self):
        """Last day of month: End + 1 falls on first of next month, covering the full month."""
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 30, 23, 59, 59))

        assert period["Start"] == "2025-11-01"
        assert period["End"] == "2025-12-01"

    def test_first_of_month_one_day_delay(self):
        """
        1-day delay on first of month: effective end_date rolls back to the last day of
        the previous month, so the full prior month is queried. Start must be before End.
        """
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=1)
        period = _capture_time_period(exporter, real_datetime(2025, 12, 1, 9, 0, 0))

        assert period["Start"] == "2025-11-01"
        assert period["End"] == "2025-12-01"
        assert period["Start"] < period["End"]

    def test_january_first_no_delay(self):
        """Year boundary edge case: Jan 1 must not produce Start == End."""
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2026, 1, 1, 0, 5, 0))

        assert period["Start"] == "2026-01-01"
        assert period["End"] == "2026-01-02"
        assert period["Start"] < period["End"]

    def test_leap_day(self):
        """Feb 29 in a leap year: Start anchors to Feb 1, End to Mar 1."""
        exporter = _make_exporter(granularity="MONTHLY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2024, 2, 29, 12, 0, 0))

        assert period["Start"] == "2024-02-01"
        assert period["End"] == "2024-03-01"


class TestDailyGranularity:
    """DAILY granularity: verify the previous-day window is always queried."""

    def test_mid_month(self):
        """Mid-month: Start is yesterday, End is today."""
        exporter = _make_exporter(granularity="DAILY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 15, 10, 0, 0))

        assert period["Start"] == "2025-11-14"
        assert period["End"] == "2025-11-15"

    def test_first_of_month_crosses_month_boundary(self):
        """First of month: Start is the last day of the previous month."""
        exporter = _make_exporter(granularity="DAILY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 1, 10, 0, 0))

        assert period["Start"] == "2025-10-31"
        assert period["End"] == "2025-11-01"

    def test_march_first_crosses_february(self):
        """March 1st (non-leap year): Start must be Feb 28."""
        exporter = _make_exporter(granularity="DAILY", data_delay_days=0)
        period = _capture_time_period(exporter, real_datetime(2025, 3, 1, 10, 0, 0))

        assert period["Start"] == "2025-02-28"
        assert period["End"] == "2025-03-01"

    def test_with_data_delay(self):
        """data_delay_days shifts the effective window back by the configured amount."""
        exporter = _make_exporter(granularity="DAILY", data_delay_days=2)
        period = _capture_time_period(exporter, real_datetime(2025, 11, 15, 10, 0, 0))

        assert period["Start"] == "2025-11-12"
        assert period["End"] == "2025-11-13"
