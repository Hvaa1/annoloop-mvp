"""Round acceptance proxies without target gold; no claim of true accuracy."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str = "manual"
    min_agreement: float = .6
    min_supported_fraction: float = .8
    max_proxy_drop: float = .03
    max_box_growth: float = 1.5
    # "trend": agreement must not fall below the selected round (no absolute floor);
    # "threshold": legacy absolute min_agreement.
    agreement_rule: str = "trend"

    def __post_init__(self):
        if self.mode not in ("manual", "automatic"):
            raise ValueError("Selection mode phải manual/automatic")
        if self.agreement_rule not in ("trend", "threshold"):
            raise ValueError("Luật agreement phải trend/threshold")
        for value in (self.min_agreement, self.min_supported_fraction, self.max_proxy_drop):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Threshold phải nằm trong [0,1]")
        if not math.isfinite(self.max_box_growth) or self.max_box_growth < 1:
            raise ValueError("Box growth phải >=1")


def validate_metrics(metrics):
    if metrics.get("domain") != "target_unlabelled_proxy":
        raise ValueError("Selection chỉ nhận proxy không gold của target")
    if metrics.get("target_gold_read") is not False:
        raise ValueError("Thiếu xác nhận không dùng target gold")
    for key in ("agreement", "supported_fraction"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Thiếu/sai proxy: " + key)
    if type(metrics.get("box_count")) is not int or metrics["box_count"] < 0:
        raise ValueError("Box count không hợp lệ")


def comparable_agreement(candidate, previous):
    return previous is not None and candidate.get("agreement_definition") == previous.get("agreement_definition")


def decide(candidate, previous, policy):
    """Gate a finished round against the selected one; proxies never measure accuracy."""
    validate_metrics(candidate)
    if previous is not None:
        validate_metrics(previous)
    reasons = []
    if candidate["box_count"] == 0:
        reasons.append("no_boxes")
    if candidate["supported_fraction"] < policy.min_supported_fraction:
        reasons.append("supported_fraction_below_threshold")
    if previous is not None and candidate["supported_fraction"] < previous["supported_fraction"] - policy.max_proxy_drop:
        reasons.append("supported_fraction_regressed")
    if policy.agreement_rule == "threshold":
        if candidate["agreement"] < policy.min_agreement:
            reasons.append("agreement_below_threshold")
        if comparable_agreement(candidate, previous) and candidate["agreement"] < previous["agreement"] - policy.max_proxy_drop:
            reasons.append("agreement_regressed")
        baseline = None
    else:
        # Trend: a round without a comparable predecessor becomes the baseline.
        baseline = comparable_agreement(candidate, previous)
        if baseline and candidate["agreement"] < previous["agreement"]:
            reasons.append("agreement_decreased")
    if previous and candidate["box_count"] > max(1, previous["box_count"]) * policy.max_box_growth:
        reasons.append("box_growth_exceeded")
    return dict(action="review" if policy.mode == "manual" else ("reject" if reasons else "accept"),
                reasons=reasons, accuracy_measured=False, agreement_rule=policy.agreement_rule,
                compared_with_previous=bool(baseline), previous_agreement=previous["agreement"] if previous else None)
