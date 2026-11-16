"""Practice dose-response under multi-source measurement error.

Every channel (self report, family report, device log) is treated as an
error-prone measurement of a learner's usual practice minutes.  No channel is
a reference standard, and the audit sample is not assumed to be a random
sample of learners.

Working model for learner ``i``, channel ``c``, occasion ``t``::

    m[i,c,t] = T[i] + a[i] + u[i,c] + e[i,c,t]

``T`` is usual minutes, ``a`` is a bias shared by all channels of a learner,
``u`` is a persistent channel-specific bias, ``e`` is transient error.  With
``R`` the average of channel means::

    Var(R)                      = Var(T) + Var(a) + Var(u) + Var(e)
    Cov(m[c1], m[c2])           = Var(T) + Var(a)      (c1 != c2)

Multichannel agreement identifies only ``S = Var(T) + Var(a)``.  The split
between true variation and common bias is not identified from agreement or
repeats alone.  The default result therefore reports a one-sided identified
bound (the side depends on the sign of the observed association) plus an
unbounded region, and keeps the common bias share as an explicit scenario.
Audit measures are used only as sensitivity anchors with a stated independence assumption; when they
disagree with the channel decomposition the result says so instead of
picking one.

The estimate is an association between usual practice and a later score
conditioned on the pre-practice baseline.  It is not a causal gain: practice
is not randomized, and neither channel agreement nor a within-household
difference removes unobserved selection.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

ANALYSIS_NAME = "dose_response"

DEFAULT_OPTIONS = {
    "seed": 20261115,
    "bootstrap_replicates": 1000,
    "common_bias_shares": [0.0, 0.25, 0.5, 0.75],
    "reference": "none",
    "support_bin_width": 20.0,
    "support_min_bin_n": 10,
}

AUDIT_ANCHORS = {
    "audit_manual_lower": "lower end of the audited manual interval",
    "audit_manual_mid": "midpoint of the audited manual interval",
    "audit_manual_upper": "upper end of the audited manual interval",
    "audit_device": "device minutes recorded by the audit",
    "audit_average": "average of the manual midpoint and audit device minutes",
}

# ---------------------------------------------------------------- utilities

def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _clean(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _weighted_cov(x, y, weight):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    total = float(weight.sum())
    if total <= 0 or len(x) < 2:
        return None
    mean_x = float((weight * x).sum() / total)
    mean_y = float((weight * y).sum() / total)
    denominator = total - float((weight * weight).sum() / total)
    if denominator <= 1e-12:
        return None
    value = float((weight * (x - mean_x) * (y - mean_y)).sum() / denominator)
    return value if math.isfinite(value) else None


def _weighted_mean(x, weight):
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)
    total = float(weight.sum())
    return float((weight * x).sum() / total) if total > 0 else None


def _ols_slope(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or float(np.ptp(x)) <= 1e-12:
        return None
    design = np.column_stack([np.ones(len(x)), x])
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    value = float(coefficient[1])
    return value if math.isfinite(value) else None


def _residualize(values, baseline):
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if not np.all(np.isfinite(values)) and np.isfinite(values).any():
        values = np.where(np.isfinite(values), values, float(np.nanmean(values)))
    if len(baseline) < 3 or not np.all(np.isfinite(baseline)) or float(np.ptp(baseline)) <= 1e-12:
        return values - values.mean()
    design = np.column_stack([np.ones(len(baseline)), baseline])
    coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficient


def _cluster_covariance(y, design, groups):
    """CR0 cluster covariance with the usual G/(G-1) correction."""
    pinv = np.linalg.pinv(design.T @ design)
    coefficient = pinv @ design.T @ y
    residual = y - design @ coefficient
    unique = list(dict.fromkeys(groups))
    meat = np.zeros((design.shape[1], design.shape[1]))
    for group in unique:
        index = np.array([position for position, value in enumerate(groups) if value == group])
        score = design[index].T @ residual[index]
        meat += np.outer(score, score)
    if len(unique) > 1:
        meat *= len(unique) / (len(unique) - 1)
    return coefficient, pinv @ meat @ pinv


def _percentile(values, confidence=0.95):
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    if len(values) < 20:
        return None
    low = (1.0 - confidence) / 2.0 * 100.0
    return [float(np.percentile(values, low)), float(np.percentile(values, 100.0 - low))]


# ---------------------------------------------------------------- preparation

def _validate_options(options):
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError("unknown options: " + ", ".join(sorted(unknown)))
    configuration = dict(DEFAULT_OPTIONS)
    configuration.update(options)

    seed = configuration["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    replicates = configuration["bootstrap_replicates"]
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 0 <= replicates <= 5000:
        raise ValueError("bootstrap_replicates must be an integer between 0 and 5000")
    width = configuration["support_bin_width"]
    if not _finite(width) or float(width) <= 0:
        raise ValueError("support_bin_width must be a positive finite number")
    width = float(width)
    minimum = configuration["support_min_bin_n"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("support_min_bin_n must be a positive integer")
    shares = configuration["common_bias_shares"]
    if not isinstance(shares, list) or not shares:
        raise ValueError("common_bias_shares must be a non-empty list")
    cleaned = []
    for share in shares:
        if not _finite(share) or not 0.0 <= float(share) < 1.0:
            raise ValueError("common_bias_shares entries must lie in [0, 1)")
        cleaned.append(float(share))
    reference = configuration["reference"]
    if not isinstance(reference, str):
        raise ValueError("reference must be a string")
    if reference != "none" and reference not in AUDIT_ANCHORS:
        raise ValueError("reference must be 'none' or one of: " + ", ".join(sorted(AUDIT_ANCHORS)))

    configuration["support_bin_width"] = width
    configuration["common_bias_shares"] = cleaned
    return configuration


def _prepare(packet):
    learners = sorted(packet["learners"], key=lambda row: row["learner"])
    identifiers = [row["learner"] for row in learners]
    metadata = {row["learner"]: row for row in learners}
    channels = list(packet["channels"])

    cells = defaultdict(list)
    for row in packet["observations"]:
        cells[(row["learner"], row["channel"], row["occasion"])].append(row["minutes"])
    observed_cells = {}
    missing_values = 0
    for key, values in cells.items():
        present = [value for value in values if value is not None]
        missing_values += len(values) - len(present)
        observed_cells[key] = float(np.mean(present)) if present else None

    occasions = defaultdict(set)
    for (learner, channel, occasion), value in observed_cells.items():
        if value is not None:
            occasions[(learner, channel)].add(occasion)
    occasions = {key: sorted(value) for key, value in occasions.items()}

    channel_mean = {}
    channel_count = {}
    for learner in identifiers:
        for channel in channels:
            values = [observed_cells[(learner, channel, occasion)]
                      for occasion in occasions.get((learner, channel), [])]
            if values:
                channel_mean.setdefault(learner, {})[channel] = float(np.mean(values))
                channel_count.setdefault(learner, {})[channel] = len(values)

    exposure = np.array([float(np.mean(list(channel_mean[learner].values()))) if learner in channel_mean
                         else float("nan") for learner in identifiers])
    means = np.full((len(identifiers), len(channels)), float("nan"))
    counts = np.zeros((len(identifiers), len(channels)))
    repeat_variance = np.full((len(identifiers), len(channels)), float("nan"))
    for position, learner in enumerate(identifiers):
        for j, channel in enumerate(channels):
            if channel in channel_mean.get(learner, {}):
                means[position, j] = channel_mean[learner][channel]
                counts[position, j] = channel_count[learner][channel]
            values = [observed_cells[(learner, channel, occasion)]
                      for occasion in occasions.get((learner, channel), [])]
            if len(values) >= 2:
                repeat_variance[position, j] = float(np.var(values, ddof=1))

    score = np.array([float(metadata[learner]["score"])
                      if _finite(metadata[learner].get("score")) else float("nan")
                      for learner in identifiers])
    baseline = np.array([float(metadata[learner]["baseline_score"])
                         if _finite(metadata[learner].get("baseline_score")) else float("nan")
                         for learner in identifiers])

    audits = {row["learner"]: row for row in packet["audits"]}
    audit_mask = np.array([learner in audits for learner in identifiers])
    probability = np.full(len(identifiers), float("nan"))
    for position, learner in enumerate(identifiers):
        if learner in audits:
            probability[position] = float(audits[learner]["selection_probability"])
    audit_weight = np.where(audit_mask, 1.0 / probability, 0.0)
    measures = {}
    for name in AUDIT_ANCHORS:
        values = np.full(len(identifiers), float("nan"))
        for position, learner in enumerate(identifiers):
            row = audits.get(learner)
            if row is None:
                continue
            interval = row.get("manual_interval")
            device = row.get("device_minutes")
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            lower, upper = float(interval[0]), float(interval[1])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                continue
            mid = (lower + upper) / 2.0
            if name == "audit_manual_lower":
                values[position] = lower
            elif name == "audit_manual_mid":
                values[position] = mid
            elif name == "audit_manual_upper":
                values[position] = upper
            elif name == "audit_device" and _finite(device):
                values[position] = float(device)
            elif name == "audit_average" and _finite(device):
                values[position] = (mid + float(device)) / 2.0
        measures[name] = values

    households = defaultdict(list)
    for position, learner in enumerate(identifiers):
        households[metadata[learner]["household"]].append(position)
    household_groups = list(households.values())
    pairs = [(members[first], members[second])
             for members in household_groups
             for first in range(len(members)) for second in range(first + 1, len(members))]

    # precomputed household transient-error products for the correlation estimate
    pair_products = []
    for first, second in pairs:
        products = []
        for channel in channels:
            first_values = {occasion: observed_cells[(identifiers[first], channel, occasion)]
                            for occasion in occasions.get((identifiers[first], channel), [])}
            second_values = {occasion: observed_cells[(identifiers[second], channel, occasion)]
                             for occasion in occasions.get((identifiers[second], channel), [])}
            shared = sorted(set(first_values) & set(second_values))
            if not shared:
                products.append(None)
                continue
            first_mean = float(np.mean(list(first_values.values())))
            second_mean = float(np.mean(list(second_values.values())))
            products.append(float(np.mean([(first_values[occasion] - first_mean) * (second_values[occasion] - second_mean)
                                           for occasion in shared])))
        pair_products.append(products)
    transient_variance = {}
    for j, channel in enumerate(channels):
        values = repeat_variance[:, j]
        values = values[np.isfinite(values)]
        transient_variance[channel] = float(np.mean(values)) if len(values) else None

    return {
        "identifiers": identifiers,
        "metadata": metadata,
        "channels": channels,
        "exposure": exposure,
        "means": means,
        "counts": counts,
        "repeat_variance": repeat_variance,
        "score": score,
        "baseline": baseline,
        "audit_mask": audit_mask,
        "audit_probability": probability,
        "audit_weight": audit_weight,
        "measures": measures,
        "household_groups": household_groups,
        "pairs": pairs,
        "pair_products": pair_products,
        "pair_clusters": [household for household, members in enumerate(household_groups)
                          for first in range(len(members)) for second in range(first + 1, len(members))],
        "transient_variance": transient_variance,
        "observed_cells": observed_cells,
        "occasions": occasions,
        "channel_mean": channel_mean,
        "channel_count": channel_count,
        "audit_count": len(audits),
        "missing_values": missing_values,
    }


# ---------------------------------------------------------------- estimation

def _fit(prepared, index, pairs, configuration):
    """All scalar quantities for one (possibly resampled) learner sample."""
    channels = prepared["channels"]
    score = prepared["score"][index]
    baseline = prepared["baseline"][index]
    exposure = prepared["exposure"][index]
    means = prepared["means"][index]
    counts = prepared["counts"][index]
    weight = prepared["audit_weight"][index]
    audited = prepared["audit_mask"][index]

    # one analysis population for association and variance decomposition:
    # finite outcome/baseline and a mean in every channel
    keep = (np.isfinite(score) & np.isfinite(baseline) & np.isfinite(exposure)
            & np.all(counts > 0, axis=1))
    positions = np.where(keep)[0]
    row_of = defaultdict(list)
    for row, local in enumerate(positions):
        row_of[int(index[local])].append(row)
    score = score[positions]
    baseline = baseline[positions]
    exposure = exposure[positions]
    means = means[positions]
    counts = counts[positions]
    weight = weight[positions]
    audited = audited[positions]

    fit = {"n": int(len(score)), "core_n": 0, "excluded_missing_channel": int((~keep).sum())}
    if len(score) < 3:
        return fit

    eta = _residualize(score, baseline)
    # partial out baseline from exposure too, so regression and variance
    # decomposition describe the same variation (Frisch-Waugh-Lovell)
    exposure_residual = _residualize(exposure, baseline)
    fit["variance_exposure"] = float(np.var(exposure_residual, ddof=1))
    fit["observed_slope"] = _ols_slope(eta, exposure_residual)

    fit["core_n"] = int(np.sum([bool(np.all(counts[row] >= 2)) for row in range(len(score))]))
    shared = None
    variance_composite = None
    if len(score) >= len(channels) + 2:
        residual_means = np.column_stack([_residualize(means[:, j], baseline)
                                          for j in range(len(channels))])
        covariance = np.cov(residual_means, rowvar=False, ddof=1)
        off = [covariance[i, j] for i in range(len(channels))
               for j in range(i + 1, len(channels))]
        if off:
            shared = float(np.mean(off))
            variance_composite = float(np.var(residual_means.mean(axis=1), ddof=1))
    fit["shared"] = shared
    fit["variance_composite"] = variance_composite
    raw_lambda = (shared / variance_composite
                  if shared is not None and shared > 0
                  and variance_composite and variance_composite > 1e-12 else None)
    fit["lambda_raw"] = raw_lambda
    fit["lambda_upper"] = min(1.0, raw_lambda) if raw_lambda is not None else None
    fit["agreement_exceeds_variance"] = bool(raw_lambda is not None and raw_lambda > 1.0 + 0.10)

    observed = fit["observed_slope"]
    lambda_upper = fit["lambda_upper"]
    fit["lower_bound"] = None
    fit["upper_bound"] = None
    if observed is not None and lambda_upper and lambda_upper > 1e-9:
        scaled = observed / lambda_upper
        # |observed| = reliability * |true| and reliability <= lambda_upper, so the
        # corrected magnitude is at least the scaled magnitude in the same direction.
        if scaled >= 0:
            fit["lower_bound"] = scaled
        else:
            fit["upper_bound"] = scaled

    scenarios = {}
    for share in configuration["common_bias_shares"]:
        slope = None
        if (observed is not None and shared is not None and shared > 0
                and variance_composite and variance_composite > 1e-12):
            reliability = min(1.0, (1.0 - share) * shared / variance_composite)
            if reliability > 1e-12:
                slope = observed / reliability
        scenarios[share] = slope
    fit["scenarios"] = scenarios

    differences_y, differences_baseline, differences_exposure, clusters = [], [], [], []
    cursor = defaultdict(int)
    for first, second, cluster in pairs:
        if not row_of.get(first) or not row_of.get(second):
            continue
        left = row_of[first][min(cursor[first], len(row_of[first]) - 1)]
        right = row_of[second][min(cursor[second], len(row_of[second]) - 1)]
        cursor[first] += 1
        cursor[second] += 1
        differences_y.append(score[left] - score[right])
        differences_baseline.append(baseline[left] - baseline[right])
        differences_exposure.append(exposure[left] - exposure[right])
        clusters.append(cluster)
    fit["within_pairs"] = len(differences_exposure)
    within = None
    if len(differences_exposure) >= 4:
        dy = np.array(differences_y, dtype=float)
        db = np.array(differences_baseline, dtype=float)
        dr = np.array(differences_exposure, dtype=float)
        if float(np.ptp(dr)) > 1e-12:
            design = (np.column_stack([np.ones(len(dr)), db, dr])
                      if float(np.ptp(db)) > 1e-12
                      else np.column_stack([np.ones(len(dr)), dr]))
            coefficient, covariance = _cluster_covariance(dy, design, clusters)
            within = float(coefficient[-1])
            variance = float(covariance[-1, -1])
            fit["within_se"] = math.sqrt(variance) if variance > 0 else None
            within_means = np.array([means[row_of[first][0]] - means[row_of[second][0]]
                                     for first, second, _ in pairs
                                     if row_of.get(first) and row_of.get(second)])
            if len(channels) > 1 and within_means.size and within_means.shape[1] == len(channels):
                off = [np.cov(within_means[:, i], within_means[:, j], ddof=1)[0, 1]
                       for i in range(len(channels)) for j in range(i + 1, len(channels))]
                shared_within = float(np.mean(off)) if off else None
                variance_within = float(np.var(dr, ddof=1))
                fit["within_shared"] = shared_within
                fit["within_variance"] = variance_within
                if shared_within is not None and shared_within > 0 and variance_within > 1e-12:
                    fit["within_lambda_upper"] = min(1.0, shared_within / variance_within)
                    bound = within / fit["within_lambda_upper"]
                    if bound >= 0:
                        fit["within_lower_bound"] = bound
                    else:
                        fit["within_upper_bound"] = bound
    fit["within_slope"] = within

    # audit anchors and the composite reliability they would imply
    anchors = {}
    for name in AUDIT_ANCHORS:
        values = prepared["measures"][name][index][positions]
        usable = audited & np.isfinite(values)
        anchor = {"slope": None, "implied_reliability": None,
                  "strength": None, "n": int(usable.sum())}
        if usable.sum() >= 6:
            selected = values[usable]
            selected_weight = weight[usable]
            covariance_exposure = _weighted_cov(exposure_residual[usable], selected, selected_weight)
            covariance_outcome = _weighted_cov(eta[usable], selected, selected_weight)
            variance_audit = _weighted_cov(exposure_residual[usable], exposure_residual[usable],
                                           selected_weight)
            if (covariance_exposure is not None and variance_audit is not None
                    and variance_audit > 1e-12):
                anchor["strength"] = covariance_exposure / variance_audit
            if covariance_exposure is not None and abs(covariance_exposure) > 1e-9:
                slope = covariance_outcome / covariance_exposure
                anchor["slope"] = slope
                if observed is not None and abs(slope) > 1e-12:
                    # IV with the anchor as instrument: the anchor's own error cancels,
                    # so the ratio is the reliability implied for the observed
                    # composite when the anchor is valid (its error independent of
                    # true minutes, the outcome error and the channel-common bias).
                    anchor["implied_reliability"] = observed / slope
        anchors[name] = anchor
    fit["anchors"] = anchors
    return fit


def _analyse(prepared, configuration):
    channels = prepared["channels"]
    index = np.arange(len(prepared["identifiers"]))
    pairs = [(first, second, cluster)
             for (first, second), cluster in zip(prepared["pairs"], prepared["pair_clusters"])]
    point = _fit(prepared, index, pairs, configuration)

    # household bootstrap
    replicates = configuration["bootstrap_replicates"]
    generator = np.random.default_rng(configuration["seed"])
    collected = defaultdict(list)
    valid = 0
    for _ in range(replicates):
        picks = generator.integers(0, len(prepared["household_groups"]), len(prepared["household_groups"]))
        sample_index = np.concatenate([prepared["household_groups"][position] for position in picks])
        sample_pairs = []
        for slot, position in enumerate(picks):
            members = prepared["household_groups"][position]
            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    sample_pairs.append((members[first], members[second], slot))
        fit = _fit(prepared, sample_index, sample_pairs, configuration)
        if fit["n"] < 3:
            continue
        valid += 1
        collected["observed_slope"].append(fit.get("observed_slope"))
        collected["lower_bound"].append(fit.get("lower_bound"))
        collected["upper_bound"].append(fit.get("upper_bound"))
        collected["within_slope"].append(fit.get("within_slope"))
        for share in configuration["common_bias_shares"]:
            collected[f"scenario:{share}"].append(fit["scenarios"].get(share))
        for name, anchor in fit["anchors"].items():
            collected[f"anchor:{name}"].append(anchor["slope"])
            collected[f"implied_reliability:{name}"].append(anchor["implied_reliability"])

    def interval(key):
        return _percentile(collected.get(key, []))

    # population and exclusions
    identifiers = prepared["identifiers"]
    missing_score = sum(1 for learner in identifiers
                        if not _finite(prepared["metadata"][learner].get("score"))
                        or not _finite(prepared["metadata"][learner].get("baseline_score")))
    no_record = sum(1 for learner in identifiers
                    if not math.isfinite(float(prepared["exposure"][identifiers.index(learner)])))
    population = {
        "learners_in_packet": len(identifiers),
        "learners_analysed": int(point["n"]),
        "households": len(prepared["household_groups"]),
        "audited_learners": prepared["audit_count"],
        "excluded": {
            "no_observed_practice_record": no_record,
            "missing_or_nonfinite_score": missing_score,
            "missing_channel_cells": int(np.isnan(prepared["means"]).sum()),
            "missing_observation_values": prepared["missing_values"],
        },
    }

    # measurement model diagnostics
    per_channel = []
    for j, channel in enumerate(channels):
        transient = prepared["transient_variance"][channel]
        repeat_counts = prepared["counts"][:, j]
        observed = repeat_counts > 0
        per_channel.append({
            "channel": channel,
            "learners_with_record": int(observed.sum()),
            "repeated_learners": int((repeat_counts >= 2).sum()),
            "transient_variance": transient,
            "repeat_agreement_sd": math.sqrt(transient) if transient else None,
            "mean_minutes": float(np.nanmean(prepared["means"][observed, j])) if observed.any() else None,
        })
    decomposition = {
        "learners_used_for_agreement": point.get("n"),
        "learners_with_two_repeats_per_channel": point.get("core_n"),
        "shared_variance_S": point.get("shared"),
        "baseline_adjusted_composite_variance": point.get("variance_composite"),
        "agreement_reliability_upper_bound": point.get("lambda_upper"),
        "agreement_exceeds_composite_variance": point.get("agreement_exceeds_variance"),
        "identified_combination": "Var(true minutes) + Var(common bias)",
        "note": ("Multichannel agreement identifies only the sum of true and common-bias "
                 "variance; it cannot show that either part is zero."),
    }

    # association
    observed_slope = point.get("observed_slope")
    observed_entry = None
    if observed_slope is not None:
        observed_entry = {
            "per_60_minutes": observed_slope * 60.0,
            "per_minute": observed_slope,
            "interval_95": (lambda values: [value * 60.0 for value in values]
                            if values else None)(interval("observed_slope")),
            "n": int(point["n"]),
            "method": "baseline-adjusted OLS of score on observed composite minutes",
            "status": "identified_association",
        }
    lower = point.get("lower_bound")
    upper = point.get("upper_bound")
    if lower is not None or upper is not None:
        identified_range = {
            "lower": lower * 60.0 if lower is not None else None,
            "upper": upper * 60.0 if upper is not None else None,
            "bound_estimate_interval_95": (lambda values: [value * 60.0 for value in values]
                                           if values else None)(
                _percentile([value for value in collected.get(
                    "lower_bound" if lower is not None else "upper_bound", [])
                    if value is not None] or [])),
            "bound_interval_note": ("Bootstrap interval for the bound estimate itself; the "
                                    "unidentified side of the association stays unbounded."),
            "status": "partially_identified",
            "reason": ("Channels share a learner-level bias that agreement cannot separate from "
                       "true variation; agreement bounds only the magnitude. The association is "
                       "bounded on one side and unbounded on the other in the direction of the "
                       "observed sign."),
            "assumptions": ("Independent transient and channel-specific errors, and a learner's "
                            "common bias is unrelated to the later score beyond true minutes. "
                            "If optimistic reporting itself predicts score, this bound can point "
                            "the wrong way; it is not assumption-free."),
        }
    else:
        if point.get("variance_exposure") is None or point.get("variance_exposure", 0) <= 1e-12:
            reason = ("Observed practice has no usable variation, so no dose-response "
                      "association is estimable.")
        elif point.get("variance_composite") is None:
            reason = ("There is no multichannel agreement information (fewer than two channels, "
                      "too few complete learners, or no exposure variation to decompose), so the "
                      "shared bias cannot be bounded.")
        else:
            reason = ("The agreement covariance is non-positive or degenerate, so the working "
                      "model does not bound the corrected association.")
        identified_range = {
            "lower": None,
            "upper": None,
            "status": "not_identified",
            "reason": reason,
            "assumptions": "None; no corrected quantity is identified.",
        }

    scenarios = []
    for share in configuration["common_bias_shares"]:
        slope = point.get("scenarios", {}).get(share)
        scenarios.append({
            "common_bias_share": share,
            "assumed_true_variance": (1.0 - share) * point["shared"]
            if point.get("shared") is not None else None,
            "reliability": (min(1.0, (1.0 - share) * point["shared"] / point["variance_composite"])
                            if point.get("shared") is not None and point.get("variance_composite")
                            else None),
            "per_60_minutes": slope * 60.0 if slope is not None else None,
            "interval_95": (lambda values: [value * 60.0 for value in values]
                            if values else None)(interval(f"scenario:{share}")),
            "status": "assumption_dependent" if slope is not None else "not_identified",
            "assumption": ("A share of the agreement variance is common bias rather than true "
                           "minutes, and that bias does not itself predict the later score. "
                           "The first part is an assumption; the second is required for the "
                           "scenario to be a dose-response association."),
        })

    within_slope = point.get("within_slope")
    within_entry = {
        "per_60_minutes": within_slope * 60.0 if within_slope is not None else None,
        "interval_95": (lambda values: [value * 60.0 for value in values]
                        if values else None)(interval("within_slope")),
        "household_pairs": int(point.get("within_pairs", 0)),
        "status": "identified_association" if within_slope is not None else "not_identified",
        "reason": ("Household members differ in practice and outcome; differences remove anything "
                   "shared inside a household, but not learner-specific selection."
                   if within_slope is not None else
                   "Too few households with two learners or no within-household exposure variation."),
        "interpretation": ("Association net of household-shared factors. It is a different "
                           "comparison from the cross-learner one, not a causal gain."),
        "within_bound_per_60": (point["within_lower_bound"] * 60.0
                                if point.get("within_lower_bound") is not None
                                else point["within_upper_bound"] * 60.0
                                if point.get("within_upper_bound") is not None else None),
        "within_bound_direction": ("lower" if point.get("within_lower_bound") is not None
                                   else "upper" if point.get("within_upper_bound") is not None
                                   else None),
    }

    anchors = []
    for name, description in AUDIT_ANCHORS.items():
        anchor_point = point.get("anchors", {}).get(name, {})
        slope = anchor_point.get("slope")
        anchors.append({
            "anchor": name,
            "description": description,
            "n": anchor_point.get("n"),
            "per_60_minutes": slope * 60.0 if slope is not None else None,
            "interval_95": (lambda values: [value * 60.0 for value in values]
                            if values else None)(interval(f"anchor:{name}")),
            "first_stage_strength": anchor_point.get("strength"),
            "implied_composite_reliability": anchor_point.get("implied_reliability"),
            "implied_composite_reliability_interval_95": interval(f"implied_reliability:{name}"),
            "status": "assumption_dependent" if slope is not None else "not_identified",
            "assumption": ("The audit measure is an instrument: its error must be independent "
                           "of true minutes, the outcome error and the channel-common bias. "
                           "That is what makes it informative; it is not assumed to be truth, "
                           "and the assumption cannot be verified from this packet."),
        })

    chosen = None
    reference = configuration["reference"]
    if reference != "none":
        anchor = point.get("anchors", {}).get(reference, {})
        if anchor.get("slope") is None:
            chosen = {"anchor": reference, "status": "not_identified",
                      "reason": "The chosen anchor has too few usable audits or no first stage."}
        else:
            draws = [value for value in collected.get(f"anchor:{reference}", [])
                     if value is not None]
            chosen = {
                "anchor": reference,
                "per_60_minutes": anchor["slope"] * 60.0,
                "interval_95": (lambda values: [value * 60.0 for value in values]
                                if values else None)(_percentile(draws)),
                "implied_composite_reliability": anchor.get("implied_reliability"),
                "status": "assumption_dependent",
                "assumption": ("The audit measure is used as an instrument for true minutes: "
                               "its error must be independent of true minutes, the outcome "
                               "error and the channel-common bias. The device is not assumed "
                               "to be truth, only to have unrelated error."),
            }

    association = {
        "observed": observed_entry,
        "identified_range": identified_range,
        "scenarios": scenarios,
        "within_household": within_entry,
        "audit_anchors": anchors,
        "chosen_reference": chosen,
        "causal_status": ("association, not a causal effect of practice on score; practice is not "
                          "randomised and selection on unobserved motivation remains"),
    }

    # support bins over observed exposure
    width = configuration["support_bin_width"]
    minimum = configuration["support_min_bin_n"]
    exposure = prepared["exposure"]
    valid_rows = np.where(np.isfinite(exposure) & np.isfinite(prepared["score"])
                          & np.isfinite(prepared["baseline"]))[0]
    bins = []
    if valid_rows.size:
        eta_all = _residualize(prepared["score"][valid_rows], prepared["baseline"][valid_rows])
        values = exposure[valid_rows]
        lower_edge = math.floor(float(values.min()) / width) * width
        upper_edge = math.ceil(float(values.max()) / width) * width
        if upper_edge <= lower_edge:
            upper_edge = lower_edge + width
        for start in np.arange(lower_edge, upper_edge, width):
            mask = (values >= start) & (values < start + width)
            count = int(mask.sum())
            if count == 0:
                continue
            selected = eta_all[mask]
            standard_error = float(selected.std(ddof=1) / math.sqrt(count)) if count >= 2 else None
            bins.append({
                "lower_minutes": float(start),
                "upper_minutes": float(start + width),
                "n": count,
                "mean_observed_minutes": float(values[mask].mean()),
                "mean_baseline_adjusted_score": float(selected.mean()),
                "standard_error": standard_error,
                "supported": count >= minimum,
            })
    supported = [entry for entry in bins if entry["supported"]]
    exposure_range = ([float(np.nanmin(exposure)), float(np.nanmax(exposure))]
                      if valid_rows.size else None)
    has_variation = (exposure_range is not None and exposure_range[1] - exposure_range[0] > 1e-9)
    support = {
        "bin_width_minutes": width,
        "minimum_bin_n": minimum,
        "observed_range": exposure_range,
        "exposure_variation": bool(has_variation),
        "bins": bins,
        "supported_intervals": ([[entry["lower_minutes"], entry["upper_minutes"]] for entry in _merge_bins(supported)]
                                if supported and has_variation else []),
        "note": ("Bins use the observed composite, which is itself error-prone; they describe where "
                 "records exist and how the baseline-adjusted score varies there, not a corrected "
                 "dose-response curve. Empty or thin bins are unsupported and must not be filled "
                 "by extrapolation. Narrow supported intervals are a data limitation, not a claim "
                 "that practice outside them has no relationship."
                 + ("" if has_variation else
                    " Every learner has the same observed exposure, so no interval of practice "
                    "intensity is supported.")),
    }

    # diagnostics
    household_error = _household_error(prepared, replicates, configuration["seed"])
    audit_selection = _audit_selection(prepared)
    audit_consistency = _audit_consistency(prepared, point)
    channel_offsets = _channel_offsets(prepared)
    degenerate = []
    if point.get("variance_exposure") is None or point.get("variance_exposure", 0) <= 1e-12:
        degenerate.append("no_exposure_variation")
    if point.get("variance_composite") is None:
        degenerate.append("no_agreement_variance")
    if prepared["audit_count"] == 0:
        degenerate.append("no_audits")
    if point.get("core_n", 0) == 0:
        degenerate.append("no_repeated_measurements")
    if point.get("within_slope") is None:
        degenerate.append("no_within_household_contrast")
    if point.get("agreement_exceeds_variance"):
        degenerate.append("agreement_exceeds_composite_variance")

    diagnostics = {
        "measurement_decomposition": decomposition,
        "household_transient_error": household_error,
        "audit_selection": audit_selection,
        "audit_consistency": audit_consistency,
        "channel_offsets": channel_offsets,
        "degenerate_conditions": degenerate,
    }

    uncertainty = {
        "method": "household-level bootstrap, percentile interval",
        "replicates_requested": replicates,
        "replicates_valid": valid,
        "seed": configuration["seed"],
        "note": ("Resampling households keeps same-household filling dependence. Intervals for "
                 "audit anchors reflect both sampling and selective auditing under the declared "
                 "probabilities, not that the anchor is unbiased."),
    }

    interpretation = _interpretation(point, population, identified_range, within_entry,
                                     anchors, degenerate, prepared["audit_count"])

    return {
        "estimand": {
            "outcome": "score after the practice period, adjusted linearly for baseline_score",
            "exposure": "usual practice minutes per learner, averaged over channels and occasions",
            "parameter": "change in baseline-adjusted score per 60 additional usual practice minutes",
            "population": "learners with a practice record and finite score and baseline_score",
            "causal_status": association["causal_status"],
        },
        "population": population,
        "measurement_model": {
            "channels": channels,
            "per_channel": per_channel,
            "decomposition": decomposition,
        },
        "association": association,
        "support": support,
        "diagnostics": diagnostics,
        "uncertainty": uncertainty,
        "interpretation": interpretation,
    }


def _merge_bins(bins):
    merged = []
    for entry in bins:
        if merged and abs(merged[-1][1] - entry["lower_minutes"]) < 1e-9:
            merged[-1] = [merged[-1][0], entry["upper_minutes"]]
        else:
            merged.append([entry["lower_minutes"], entry["upper_minutes"]])
    return [{"lower_minutes": lower, "upper_minutes": upper} for lower, upper in merged]


def _household_error(prepared, replicates, seed):
    channels = prepared["channels"]
    products = prepared["pair_products"]
    transient = prepared["transient_variance"]
    if not products:
        return {"pair_count": 0, "correlation": None, "interval_95": None,
                "valid_replicates": 0, "note": "No households with two learners."}

    def estimate(rows):
        values = []
        for j in range(len(channels)):
            column = [row[j] for row in rows if row[j] is not None]
            denominator = transient.get(channels[j])
            if column and denominator and denominator > 1e-12:
                values.append(float(np.mean(column)) / denominator)
        return float(np.mean(values)) if values else None

    point = estimate(products)
    draws = []
    if replicates > 0 and len(products) >= 4:
        generator = np.random.default_rng(seed)
        for _ in range(replicates):
            positions = generator.integers(0, len(products), len(products))
            value = estimate([products[position] for position in positions])
            if value is not None and math.isfinite(value):
                draws.append(value)
    return {
        "pair_count": len(products),
        "correlation": point,
        "interval_95": _percentile(draws),
        "valid_replicates": len(draws),
        "note": ("Correlation of transient errors between two learners in one household. "
                 "Positive values mean one person's filling slips move together across learners, "
                 "so repeats are not independent information; the household bootstrap keeps "
                 "this clustering."),
    }


def _audit_selection(prepared):
    mask = prepared["audit_mask"]
    total = len(mask)
    selected = int(mask.sum())
    if selected == 0:
        return {
            "audited": 0,
            "fraction_of_learners": 0.0,
            "declared_probabilities": [],
            "note": ("No audit records. Multichannel repeats still bound the association but "
                     "cannot reveal a common bias shared by every channel."),
        }
    probabilities = prepared["audit_probability"][mask]
    weight = prepared["audit_weight"][mask]
    exposure = prepared["exposure"]
    complete = np.isfinite(exposure) & np.isfinite(prepared["score"]) & np.isfinite(prepared["baseline"])
    cohort_exposure = exposure[complete]
    selected_exposure = exposure[mask & complete]
    selected_weight = prepared["audit_weight"][mask & complete]
    eta_all = np.full(len(prepared["identifiers"]), float("nan"))
    usable_outcome = np.isfinite(prepared["score"]) & np.isfinite(prepared["baseline"])
    if usable_outcome.sum() >= 3:
        eta_all[usable_outcome] = _residualize(prepared["score"][usable_outcome],
                                               prepared["baseline"][usable_outcome])
    strata = {}
    for value in sorted(set(float(item) for item in probabilities)):
        rows = np.isclose(prepared["audit_probability"], value) & mask
        usable = rows & np.isfinite(exposure) & np.isfinite(eta_all)
        strata[str(value)] = {
            "audited": int(rows.sum()),
            "mean_observed_minutes": (float(np.mean(exposure[usable])) if usable.any() else None),
            "mean_baseline_adjusted_score": (float(np.mean(eta_all[usable])) if usable.any() else None),
        }
    unweighted = float(np.mean(selected_exposure)) if selected_exposure.size else None
    weighted = _weighted_mean(selected_exposure, selected_weight) if selected_weight.sum() > 0 else None
    overall = float(np.mean(cohort_exposure)) if cohort_exposure.size else None
    return {
        "audited": selected,
        "fraction_of_learners": selected / total if total else None,
        "declared_probabilities": sorted(set(float(item) for item in probabilities)),
        "effective_sample_size": float(weight.sum() ** 2 / (weight * weight).sum()),
        "sum_of_inverse_probability_weights": float(weight.sum()),
        "mean_observed_minutes": {
            "audit_unweighted": unweighted,
            "audit_inverse_probability_weighted": weighted,
            "cohort": overall,
        },
        "strata": strata,
        "note": ("Audits are a probability sample with declared selection probabilities that vary "
                 "with observed practice. Anchors use inverse-probability weights; an unweighted "
                 "audit average is not a population average. If the declared probabilities do not "
                 "describe the actual selection, anchor estimates inherit that bias."),
    }


def _audit_consistency(prepared, point):
    anchors = point.get("anchors", {})
    relative = {name: entry["implied_reliability"] for name, entry in anchors.items()
                if entry.get("implied_reliability") is not None}
    measures = prepared["measures"]
    mask = prepared["audit_mask"]
    coverage = {}
    if mask.any():
        low = measures["audit_manual_lower"][mask]
        high = measures["audit_manual_upper"][mask]
        coverage["observed_composite_inside_manual_interval"] = float(
            np.mean((prepared["exposure"][mask] >= low) & (prepared["exposure"][mask] <= high)))
        coverage["audit_device_inside_manual_interval"] = float(
            np.mean((measures["audit_device"][mask] >= low) & (measures["audit_device"][mask] <= high)))
        mid = measures["audit_manual_mid"][mask]
        device = measures["audit_device"][mask]
        correlation = None
        if mask.sum() >= 3 and np.std(mid) > 1e-12 and np.std(device) > 1e-12:
            correlation = float(np.corrcoef(mid, device)[0, 1])
        coverage["manual_device_correlation"] = correlation
        coverage["mean_device_minus_manual_mid"] = float(np.mean(device - mid))
        widths = high - low
        coverage["manual_interval_width_mean"] = float(np.mean(widths)) if widths.size else None
    spread = None
    if relative:
        values = list(relative.values())
        spread = {"minimum": float(min(values)), "maximum": float(max(values)),
                  "range": float(max(values) - min(values))}
    check = None
    lambda_upper = point.get("lambda_upper")
    if relative and lambda_upper is not None:
        implied = float(np.mean(list(relative.values())))
        # Under anchor validity the implied composite reliability cannot exceed the
        # agreement bound; a violation means the anchor shares the common bias.
        violates = implied > lambda_upper + 0.05 or implied < -0.05
        if violates:
            message = ("The audit measure implies composite reliability above what agreement "
                       "allows. Either the anchor error is not independent of the "
                       "channel-common bias, or the working model fails; neither source "
                       "should be treated as the correction.")
        elif lambda_upper - implied > 0.2:
            message = ("The audit measure implies composite reliability well below the "
                       "agreement-only upper bound. Agreement alone cannot rule out that much "
                       "common bias, so the anchor scenario and the agreement bound must both "
                       "stay visible.")
        else:
            message = ("The audit measure is compatible with the agreement bound. Agreement "
                       "alone still does not identify the split, so any anchor-based correction "
                       "remains an assumption rather than a default.")
        check = {
            "agreement_reliability_upper_bound": lambda_upper,
            "audit_implied_composite_reliability": implied,
            "incompatible_with_working_model": bool(violates),
            "message": message,
        }
    return {
        "implied_composite_reliability_by_measure": relative,
        "implied_reliability_spread": spread,
        "interval_coverage": coverage,
        "reliability_check": check,
        "note": ("Manual and device audit measures correlating with each other is not evidence that "
                 "either is accurate: one reporter or one device can share the same bias. "
                 "Anchors are sensitivity evidence, not ground truth."),
    }


def _channel_offsets(prepared):
    channels = prepared["channels"]
    means = prepared["means"]
    offsets = {}
    for j, channel in enumerate(channels):
        values = means[:, j]
        present = np.isfinite(values)
        offsets[channel] = {
            "mean_minutes": float(np.nanmean(values)) if present.any() else None,
            "learners": int(present.sum()),
        }
    differences = []
    for first in range(len(channels)):
        for second in range(first + 1, len(channels)):
            pair = means[:, first] - means[:, second]
            pair = pair[np.isfinite(pair)]
            if pair.size:
                differences.append({
                    "channels": [channels[first], channels[second]],
                    "mean_difference_minutes": float(pair.mean()),
                    "sd_difference_minutes": float(pair.std(ddof=1)) if pair.size > 1 else None,
                    "learners": int(pair.size),
                })
    return {
        "means": offsets,
        "pairwise_differences": differences,
        "note": ("Channel offsets describe reporting patterns, not accuracy. A persistent offset "
                 "is measurement bias, not a practice difference."),
    }


def _format(value, digits=2):
    return "not identified" if value is None else f"{value:.{digits}f}"


def _interpretation(point, population, identified_range, within_entry, anchors, degenerate, audit_count):
    notes = []
    observed = point.get("observed_slope")
    if observed is None:
        notes.append("The observed practice records show no usable exposure variation, so no "
                     "dose-response association is estimated.")
    else:
        notes.append(f"Baseline-adjusted association is {_format(observed * 60.0)} score points per "
                     f"60 usual practice minutes in the {population['learners_analysed']} learners with records.")
    lower = identified_range.get("lower")
    upper = identified_range.get("upper")
    if lower is not None:
        notes.append(f"Because channels share an unidentified learner-level bias, the data identify "
                     f"a corrected association of at least {_format(lower)} points per 60 minutes "
                     f"in the observed direction; the other side is unbounded.")
    elif upper is not None:
        notes.append(f"Because channels share an unidentified learner-level bias, the data identify "
                     f"a corrected association of at most {_format(upper)} points per 60 minutes "
                     f"in the observed direction; the other side is unbounded.")
    else:
        notes.append("The corrected association is not bounded by these records.")
    if point.get("shared") is not None and point.get("variance_composite") is not None:
        notes.append("Multichannel agreement accounts for at most "
                     f"{_format(point['lambda_upper'])} of the composite variance; it cannot show "
                     "that the shared part is true practice rather than a common bias.")
    usable_anchors = [anchor for anchor in anchors if anchor.get("per_60_minutes") is not None]
    if not audit_count:
        notes.append("No audit records exist, so there is no independent anchor to challenge a "
                     "common bias shared by every channel.")
    elif usable_anchors:
        ratios = [anchor["implied_composite_reliability"]
                  for anchor in usable_anchors
                  if anchor.get("implied_composite_reliability") is not None]
        if ratios:
            notes.append(f"Under independent anchor errors, audit measures imply composite "
                         f"reliability between {_format(min(ratios))} and {_format(max(ratios))}. "
                         "This is a sensitivity check, not a correction of record; the audit "
                         "measure is not truth either.")
        notes.append("The audit sample is a probability sample; inverse-probability weighting is used, "
                     "and audit anchors are reported as scenarios rather than substituted for the "
                     "records.")
    if within_entry.get("per_60_minutes") is not None:
        notes.append("Within-household differences give a separate association of "
                     f"{_format(within_entry['per_60_minutes'])} points per 60 minutes; this removes "
                     "household-shared factors but is not a causal gain.")
    else:
        notes.append("No within-household contrast is available; household-shared bias cannot be "
                     "differenced out.")
    notes.append("Practice is not randomised, so none of these quantities is a causal return to "
                 "extra practice; they are associations that differ in which biases they remove.")
    if "no_repeated_measurements" in degenerate:
        notes.append("No learner has two or more repeats per channel, so transient error cannot be "
                     "estimated from repeats. Multichannel agreement still bounds the shared "
                     "variance, but the correction stays one-sided and assumption-free only in "
                     "that bound.")
    if "agreement_exceeds_composite_variance" in degenerate:
        notes.append("The agreement covariance exceeds the total composite variance, which means the "
                     "working independence model does not fit; do not correct by a fixed factor.")
    return {
        "summary": notes,
        "not_a_causal_effect": True,
        "default_resolution": ("Uncertainty between common bias and true variation is retained; "
                               "the default does not pick a correction factor."),
    }


def dose_response(packet, options=None):
    configuration = _validate_options(options)
    prepared = _prepare(packet)
    result = _analyse(prepared, configuration)
    return _clean(result)

