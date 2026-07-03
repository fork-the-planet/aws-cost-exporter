"""Tests for metric_types (list) support added in Issue #35."""

from unittest.mock import MagicMock, call, patch

import pytest

from app.exporter import METRIC_TYPE_LABEL, MetricExporter

_GROUP_BY_DISABLED = {"enabled": False, "groups": [], "merge_minor_cost": {"enabled": False}}
_GROUP_BY_BY_SERVICE = {
    "enabled": True,
    "groups": [{"type": "DIMENSION", "key": "SERVICE", "label_name": "ServiceName"}],
    "merge_minor_cost": {"enabled": False, "threshold": 0, "tag_value": "other"},
}
_TARGETS = [{"Publisher": "123456789012"}]


@pytest.fixture(autouse=True)
def _mock_gauge(monkeypatch):
    # Use a lambda factory so Gauge(name, desc, labels) returns an unconstrained MagicMock.
    # Passing MagicMock directly would make the first positional arg (metric_name) the `spec`,
    # locking the returned mock to string attributes only (no .labels).
    monkeypatch.setattr("app.exporter.Gauge", lambda *args, **kwargs: MagicMock())


def _make_exporter(metric_type=None, metric_types=None, group_by=None):
    return MetricExporter(
        polling_interval_seconds=3600,
        metric_name="test_metric",
        aws_access_key="",
        aws_access_secret="",
        aws_assumed_role_name="",
        group_by=group_by or _GROUP_BY_DISABLED,
        targets=_TARGETS,
        metric_type=metric_type,
        metric_types=metric_types,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_single_metric_type_no_label(self):
        exporter = _make_exporter(metric_type="UnblendedCost")

        assert exporter.metric_types == ["UnblendedCost"]
        assert exporter.add_metric_type_label is False
        assert METRIC_TYPE_LABEL not in exporter.labels

    def test_metric_types_list_adds_label(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"])

        assert exporter.metric_types == ["AmortizedCost", "UnblendedCost"]
        assert exporter.add_metric_type_label is True
        assert METRIC_TYPE_LABEL in exporter.labels

    def test_metric_types_list_is_copied(self):
        original = ["AmortizedCost", "UnblendedCost"]
        exporter = _make_exporter(metric_types=original)
        original.append("BlendedCost")

        assert exporter.metric_types == ["AmortizedCost", "UnblendedCost"]

    def test_single_metric_type_default_record_types(self):
        exporter = _make_exporter(metric_type="UnblendedCost")

        assert exporter.record_types == ["Usage"]

    def test_amortized_in_metric_types_uses_amortized_record_types(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"])

        assert "SavingsPlanCoveredUsage" in exporter.record_types
        assert "DiscountedUsage" in exporter.record_types

    def test_no_amortized_in_metric_types_uses_base_record_types(self):
        exporter = _make_exporter(metric_types=["UnblendedCost", "BlendedCost"])

        assert exporter.record_types == ["Usage"]


# ---------------------------------------------------------------------------
# AWS Cost Explorer API call — Metrics parameter
# ---------------------------------------------------------------------------


class TestCeApiMetricsParam:
    def _run_query(self, exporter, result_payload):
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = result_payload
        exporter.query_aws_cost_explorer(mock_client, _GROUP_BY_DISABLED)
        return mock_client.get_cost_and_usage.call_args

    def test_single_metric_type_sends_single_element_list(self):
        exporter = _make_exporter(metric_type="UnblendedCost")
        call_args = self._run_query(exporter, {"ResultsByTime": []})

        assert call_args.kwargs["Metrics"] == ["UnblendedCost"]

    def test_multiple_metric_types_sent_in_one_api_call(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"])
        call_args = self._run_query(exporter, {"ResultsByTime": []})

        assert call_args.kwargs["Metrics"] == ["AmortizedCost", "UnblendedCost"]

    def test_three_metric_types_sent_in_one_api_call(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost", "BlendedCost"])
        call_args = self._run_query(
            exporter,
            {"ResultsByTime": []},
        )

        assert call_args.kwargs["Metrics"] == ["AmortizedCost", "UnblendedCost", "BlendedCost"]


# ---------------------------------------------------------------------------
# Metric label emission — no group_by
# ---------------------------------------------------------------------------


class TestMetricLabelsNoGroupBy:
    def _build_ce_response(self, *metric_types):
        """Build a fake CE ResultsByTime entry with $10 for each metric type."""
        totals = {mt: {"Amount": "10.00", "Unit": "USD"} for mt in metric_types}
        return {"ResultsByTime": [{"Total": totals, "Groups": [], "Estimated": False}]}

    def test_single_metric_type_no_metric_type_label(self):
        exporter = _make_exporter(metric_type="UnblendedCost")
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_ce_response("UnblendedCost")

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        label_call_kwargs = exporter.cost_metric.labels.call_args_list[0][1]
        assert METRIC_TYPE_LABEL not in label_call_kwargs

    def test_multiple_metric_types_each_gets_metric_type_label(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"])
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_ce_response("AmortizedCost", "UnblendedCost")

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        all_kwargs = [c[1] for c in exporter.cost_metric.labels.call_args_list]
        metric_type_values = {kw[METRIC_TYPE_LABEL] for kw in all_kwargs}
        assert metric_type_values == {"AmortizedCost", "UnblendedCost"}

    def test_multiple_metric_types_correct_values_set(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"])
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {
                    "Total": {
                        "AmortizedCost": {"Amount": "42.00", "Unit": "USD"},
                        "UnblendedCost": {"Amount": "39.50", "Unit": "USD"},
                    },
                    "Groups": [],
                    "Estimated": False,
                }
            ]
        }

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        set_calls = exporter.cost_metric.labels.return_value.set.call_args_list
        values = {c[0][0] for c in set_calls}
        assert 42.0 in values
        assert 39.5 in values


# ---------------------------------------------------------------------------
# Metric label emission — with group_by
# ---------------------------------------------------------------------------


class TestMetricLabelsWithGroupBy:
    def _build_ce_response_grouped(self, service, *metric_types):
        metrics = {mt: {"Amount": "5.00", "Unit": "USD"} for mt in metric_types}
        return {
            "ResultsByTime": [
                {
                    "Total": {},
                    "Groups": [{"Keys": [service], "Metrics": metrics}],
                    "Estimated": False,
                }
            ]
        }

    def test_grouped_multiple_types_each_group_item_gets_metric_type_label(self):
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"], group_by=_GROUP_BY_BY_SERVICE)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_ce_response_grouped(
            "AmazonEC2", "AmortizedCost", "UnblendedCost"
        )

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        all_kwargs = [c[1] for c in exporter.cost_metric.labels.call_args_list]
        assert len(all_kwargs) == 2
        metric_type_values = {kw[METRIC_TYPE_LABEL] for kw in all_kwargs}
        assert metric_type_values == {"AmortizedCost", "UnblendedCost"}

    def test_grouped_single_type_no_metric_type_label(self):
        exporter = _make_exporter(metric_type="UnblendedCost", group_by=_GROUP_BY_BY_SERVICE)
        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = self._build_ce_response_grouped("AmazonEC2", "UnblendedCost")

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        label_kwargs = exporter.cost_metric.labels.call_args_list[0][1]
        assert METRIC_TYPE_LABEL not in label_kwargs


# ---------------------------------------------------------------------------
# Merge minor cost — tracked per metric_type
# ---------------------------------------------------------------------------


class TestMergeMinorCostPerType:
    _GROUP_BY_WITH_MERGE = {
        "enabled": True,
        "groups": [{"type": "DIMENSION", "key": "SERVICE", "label_name": "ServiceName"}],
        "merge_minor_cost": {"enabled": True, "threshold": 10.0, "tag_value": "other"},
    }

    def test_merge_minor_cost_applied_per_metric_type(self):
        """Minor cost items should be merged independently per metric type."""
        exporter = _make_exporter(metric_types=["AmortizedCost", "UnblendedCost"], group_by=self._GROUP_BY_WITH_MERGE)
        mock_client = MagicMock()
        # AmazonEC2 has high Amortized ($50) but low Unblended ($3) → only Unblended merges
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {
                    "Total": {},
                    "Groups": [
                        {
                            "Keys": ["AmazonEC2"],
                            "Metrics": {
                                "AmortizedCost": {"Amount": "50.00", "Unit": "USD"},
                                "UnblendedCost": {"Amount": "3.00", "Unit": "USD"},
                            },
                        }
                    ],
                    "Estimated": False,
                }
            ]
        }

        with patch.object(exporter, "_get_aws_client", return_value=mock_client):
            exporter._fetch_with_filters(_TARGETS[0], [], {})

        all_kwargs = [c[1] for c in exporter.cost_metric.labels.call_args_list]
        # AmortizedCost for EC2 should be set directly (not merged)
        amortized_direct = [kw for kw in all_kwargs if kw.get(METRIC_TYPE_LABEL) == "AmortizedCost" and kw.get("ServiceName") == "AmazonEC2"]
        assert len(amortized_direct) == 1

        # UnblendedCost for EC2 should be merged into "other"
        unblended_ec2 = [kw for kw in all_kwargs if kw.get(METRIC_TYPE_LABEL) == "UnblendedCost" and kw.get("ServiceName") == "AmazonEC2"]
        assert len(unblended_ec2) == 0

        unblended_other = [kw for kw in all_kwargs if kw.get(METRIC_TYPE_LABEL) == "UnblendedCost" and kw.get("ServiceName") == "other"]
        assert len(unblended_other) == 1
