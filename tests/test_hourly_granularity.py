"""
Tests for HOURLY granularity support (PR #39).

Covers:
- UTC top-of-hour alignment of the query window in query_aws_cost_explorer()
- hourly_time_range_hours (and data_delay_days) -> Start/End window computation
- PeriodStart label emitted once per hourly bucket, absent for DAILY/MONTHLY
- Config validation bounds for hourly_time_range_hours, including the combined
  constraint hourly_time_range_hours + data_delay_days * 24 <= 336
"""

from datetime import datetime as real_datetime
from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest

from app.exporter import MetricExporter
from main import validate_configs

_GROUP_BY_DISABLED = {
    "enabled": False,
    "groups": [],
    "merge_minor_cost": {"enabled": False},
}
_GROUP_BY_BY_SERVICE = {
    "enabled": True,
    "groups": [{"type": "DIMENSION", "key": "SERVICE", "label_name": "ServiceName"}],
    "merge_minor_cost": {"enabled": False, "threshold": 0, "tag_value": "other"},
}
_TARGETS = [{"Publisher": "123456789012"}]


@pytest.fixture(autouse=True)
def _mock_gauge(monkeypatch):
    # Use a lambda factory so Gauge(name, desc, labels) returns an unconstrained MagicMock.
    monkeypatch.setattr("app.exporter.Gauge", lambda *args, **kwargs: MagicMock())


def _make_exporter(granularity="HOURLY", hourly_time_range_hours=24, data_delay_days=0, group_by=None):
    return MetricExporter(
        polling_interval_seconds=3600,
        metric_name="test_hourly_metric",
        aws_access_key="",
        aws_access_secret="",
        aws_assumed_role_name="",
        group_by=group_by or _GROUP_BY_DISABLED,
        targets=_TARGETS,
        metric_type="UnblendedCost",
        granularity=granularity,
        hourly_time_range_hours=hourly_time_range_hours,
        data_delay_days=data_delay_days,
    )


def _capture_time_period(exporter, now_utc):
    """
    Invoke query_aws_cost_explorer with a fixed datetime.now(timezone.utc) and
    return the TimePeriod dict that was passed to GetCostAndUsage.
    """
    mock_client = MagicMock()
    mock_client.get_cost_and_usage.return_value = {"ResultsByTime": []}

    with patch("app.exporter.datetime") as mock_dt:
        mock_dt.now.return_value = now_utc
        mock_dt.today.return_value = now_utc.replace(tzinfo=None)
        mock_dt.side_effect = lambda *args, **kwargs: real_datetime(*args, **kwargs)
        exporter.query_aws_cost_explorer(mock_client, _GROUP_BY_DISABLED)

    return mock_client.get_cost_and_usage.call_args.kwargs["TimePeriod"]


# ---------------------------------------------------------------------------
# Query window computation
# ---------------------------------------------------------------------------


class TestHourlyTimeWindow:
    """HOURLY granularity: verify the UTC-aligned Start/End window."""

    def test_end_aligned_to_top_of_hour_utc(self):
        """End must be the current UTC hour with minutes/seconds truncated."""
        exporter = _make_exporter(hourly_time_range_hours=2)
        period = _capture_time_period(exporter, real_datetime(2026, 7, 15, 10, 47, 23, tzinfo=timezone.utc))

        assert period["End"] == "2026-07-15T10:00:00Z"
        assert period["Start"] == "2026-07-15T08:00:00Z"

    def test_window_spans_hourly_time_range_hours(self):
        """Start must be exactly hourly_time_range_hours before the aligned End."""
        exporter = _make_exporter(hourly_time_range_hours=24)
        period = _capture_time_period(exporter, real_datetime(2026, 7, 15, 0, 5, 0, tzinfo=timezone.utc))

        assert period["End"] == "2026-07-15T00:00:00Z"
        assert period["Start"] == "2026-07-14T00:00:00Z"

    def test_window_crosses_month_boundary(self):
        """A window reaching into the previous month must roll the date correctly."""
        exporter = _make_exporter(hourly_time_range_hours=6)
        period = _capture_time_period(exporter, real_datetime(2026, 8, 1, 3, 30, 0, tzinfo=timezone.utc))

        assert period["Start"] == "2026-07-31T21:00:00Z"
        assert period["End"] == "2026-08-01T03:00:00Z"

    def test_data_delay_days_shifts_window_back(self):
        """data_delay_days must shift both Start and End into the past by whole days."""
        exporter = _make_exporter(hourly_time_range_hours=24, data_delay_days=1)
        period = _capture_time_period(exporter, real_datetime(2026, 7, 15, 10, 47, 23, tzinfo=timezone.utc))

        assert period["End"] == "2026-07-14T10:00:00Z"
        assert period["Start"] == "2026-07-13T10:00:00Z"

    def test_uses_utc_clock_not_local_today(self):
        """The hourly window must come from datetime.now(timezone.utc), not datetime.today()."""
        exporter = _make_exporter(hourly_time_range_hours=1)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = {"ResultsByTime": []}

        with patch("app.exporter.datetime") as mock_dt:
            mock_dt.now.return_value = real_datetime(2026, 7, 15, 10, 47, 23, tzinfo=timezone.utc)
            # Simulate a local clock far away from UTC; it must not leak into the window
            mock_dt.today.return_value = real_datetime(2026, 7, 15, 22, 47, 23)
            mock_dt.side_effect = lambda *args, **kwargs: real_datetime(*args, **kwargs)
            exporter.query_aws_cost_explorer(mock_client, _GROUP_BY_DISABLED)

        period = mock_client.get_cost_and_usage.call_args.kwargs["TimePeriod"]
        assert period["End"] == "2026-07-15T10:00:00Z"
        mock_dt.now.assert_called_once_with(timezone.utc)


# ---------------------------------------------------------------------------
# PeriodStart label emission
# ---------------------------------------------------------------------------


class TestPeriodStartLabel:
    def _build_hourly_response(self, *period_starts):
        """
        Build a fake CE response with one grouped $5 datapoint per hourly bucket.

        Note: this helper only produces *grouped* responses (each entry has
        ``Groups`` and an empty ``Total``), so it is only valid for exporters with
        ``group_by`` enabled. The non-grouped code path reads ``result["Total"]``
        and would KeyError on this shape.

        ``End`` is set to a sentinel that differs from every ``Start`` so the tests
        prove the ``PeriodStart`` label is sourced from ``Start`` (not ``End``).
        """
        return {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": start, "End": "2999-01-01T00:00:00Z"},
                    "Total": {},
                    "Groups": [
                        {"Keys": ["AmazonEC2"], "Metrics": {"UnblendedCost": {"Amount": "5.00", "Unit": "USD"}}}
                    ],
                    "Estimated": False,
                }
                for start in period_starts
            ]
        }

    def test_hourly_adds_period_start_to_gauge_labels(self):
        exporter = _make_exporter(granularity="HOURLY")
        assert "PeriodStart" in exporter.labels

    def test_daily_and_monthly_do_not_add_period_start_to_gauge_labels(self):
        for granularity in ("DAILY", "MONTHLY"):
            exporter = _make_exporter(granularity=granularity)
            assert "PeriodStart" not in exporter.labels

    def test_hourly_emits_period_start_per_bucket(self):
        """Each hourly ResultsByTime entry must become its own series, keyed by PeriodStart."""
        exporter = _make_exporter(granularity="HOURLY", group_by=_GROUP_BY_BY_SERVICE)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_hourly_response(
            "2026-07-15T08:00:00Z", "2026-07-15T09:00:00Z", "2026-07-15T10:00:00Z"
        )

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        all_kwargs = [c[1] for c in exporter.cost_metric.labels.call_args_list]
        assert len(all_kwargs) == 3
        assert {kw["PeriodStart"] for kw in all_kwargs} == {
            "2026-07-15T08:00:00Z",
            "2026-07-15T09:00:00Z",
            "2026-07-15T10:00:00Z",
        }

    def test_hourly_merged_minor_cost_keeps_period_start(self):
        """The merge_minor_cost catch-all series must also carry the bucket's PeriodStart."""
        group_by = {
            "enabled": True,
            "groups": [{"type": "DIMENSION", "key": "SERVICE", "label_name": "ServiceName"}],
            "merge_minor_cost": {"enabled": True, "threshold": 10.0, "tag_value": "other"},
        }
        exporter = _make_exporter(granularity="HOURLY", group_by=group_by)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_hourly_response(
            "2026-07-15T08:00:00Z", "2026-07-15T09:00:00Z"
        )

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        all_kwargs = [c[1] for c in exporter.cost_metric.labels.call_args_list]
        merged = [kw for kw in all_kwargs if kw.get("ServiceName") == "other"]
        assert {kw["PeriodStart"] for kw in merged} == {"2026-07-15T08:00:00Z", "2026-07-15T09:00:00Z"}

    def test_daily_does_not_emit_period_start(self):
        exporter = _make_exporter(granularity="DAILY", group_by=_GROUP_BY_BY_SERVICE)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_hourly_response("2026-07-14")

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        label_kwargs = exporter.cost_metric.labels.call_args_list[0][1]
        assert "PeriodStart" not in label_kwargs

    def test_monthly_does_not_emit_period_start(self):
        exporter = _make_exporter(granularity="MONTHLY", group_by=_GROUP_BY_BY_SERVICE)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_hourly_response("2026-07-01")

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        label_kwargs = exporter.cost_metric.labels.call_args_list[0][1]
        assert "PeriodStart" not in label_kwargs


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _make_config(**metric_overrides):
    metric = {
        "metric_name": "test_metric",
        "metric_type": "UnblendedCost",
        "group_by": {"enabled": False},
        "granularity": "HOURLY",
    }
    metric.update(metric_overrides)
    return {"target_aws_accounts": [{"Publisher": "123456789012"}], "metrics": [metric]}


class TestHourlyTimeRangeValidation:
    def test_missing_defaults_to_24(self, caplog):
        config = _make_config()
        validate_configs(config)

        assert config["metrics"][0]["hourly_time_range_hours"] == 24
        assert "defaulting to 24" in caplog.text

    @pytest.mark.parametrize("valid_value", [1, 24, 336])
    def test_valid_bounds_accepted(self, valid_value):
        config = _make_config(hourly_time_range_hours=valid_value)
        validate_configs(config)

        assert config["metrics"][0]["hourly_time_range_hours"] == valid_value

    @pytest.mark.parametrize("invalid_value", [0, -1, 337, True, "24", 24.5, None])
    def test_invalid_values_rejected(self, invalid_value):
        config = _make_config(hourly_time_range_hours=invalid_value)

        with pytest.raises(SystemExit) as exc_info:
            validate_configs(config)
        assert exc_info.value.code == 1

    def test_ignored_for_daily_granularity(self):
        """DAILY metrics must not require or default hourly_time_range_hours."""
        config = _make_config(granularity="DAILY")
        validate_configs(config)

        assert "hourly_time_range_hours" not in config["metrics"][0]

    def test_set_on_non_hourly_metric_warns(self, caplog):
        """Setting hourly_time_range_hours on a non-HOURLY metric warns but does not fail."""
        config = _make_config(granularity="DAILY", hourly_time_range_hours=48)
        validate_configs(config)

        assert "will be ignored" in caplog.text
        # The value is left untouched (not validated against the 1..336 bounds).
        assert config["metrics"][0]["hourly_time_range_hours"] == 48

    def test_no_warning_when_hourly(self, caplog):
        """A valid HOURLY metric must not emit the ignored-value warning."""
        config = _make_config(granularity="HOURLY", hourly_time_range_hours=48)
        validate_configs(config)

        assert "will be ignored" not in caplog.text


class TestHourlyRetentionWithDataDelay:
    """hourly_time_range_hours + data_delay_days * 24 must stay within AWS's 336h retention."""

    def test_max_range_with_zero_delay_accepted(self):
        validate_configs(_make_config(hourly_time_range_hours=336, data_delay_days=0))

    def test_max_range_with_delay_rejected(self):
        with pytest.raises(SystemExit) as exc_info:
            validate_configs(_make_config(hourly_time_range_hours=336, data_delay_days=1))
        assert exc_info.value.code == 1

    def test_combined_exactly_at_limit_accepted(self):
        # 312 + 1 * 24 == 336
        validate_configs(_make_config(hourly_time_range_hours=312, data_delay_days=1))

    def test_combined_one_hour_over_limit_rejected(self):
        # 313 + 1 * 24 == 337
        with pytest.raises(SystemExit) as exc_info:
            validate_configs(_make_config(hourly_time_range_hours=313, data_delay_days=1))
        assert exc_info.value.code == 1

    def test_default_range_with_delay_within_limit_accepted(self):
        # default 24 + 2 * 24 == 72
        config = _make_config(data_delay_days=2)
        validate_configs(config)

        assert config["metrics"][0]["hourly_time_range_hours"] == 24

    def test_delay_alone_does_not_affect_daily(self):
        """The combined constraint only applies to HOURLY metrics."""
        validate_configs(_make_config(granularity="DAILY", data_delay_days=30))
