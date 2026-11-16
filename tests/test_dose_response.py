import copy
import random
import unittest

import numpy as np

from app.catalog import local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze
from app.protocol import digest


def _packet(learners, observations, audits, revision="synthetic-test", channels=("self", "family", "device")):
    packet = {
        "revision": revision,
        "synthetic": True,
        "channels": list(channels),
        "learners": learners,
        "observations": observations,
        "audits": audits,
    }
    validate(packet)
    return packet


def _synthetic(seed=7, learners=120, true_effect_per_minute=0.1, common_bias_sd=12.0,
               transient_sd=5.0, channel_bias=(0.0, 4.0, -6.0), audit=True,
               selection_depends_on_practice=True):
    """Small deterministic DGP with known truth; channels are all error-prone."""
    generator = np.random.default_rng(seed)
    identifiers = [f"L{i:03d}" for i in range(learners)]
    households = [f"h{i // 2:03d}" for i in range(learners)]
    truth = generator.uniform(30.0, 110.0, learners)
    common = generator.normal(0.0, common_bias_sd, learners)
    baseline = generator.normal(50.0, 10.0, learners)
    score = baseline + true_effect_per_minute * truth + generator.normal(0.0, 3.0, learners)

    channels = ["self", "family", "device"]
    observations = []
    channel_means = {}
    for index, learner in enumerate(identifiers):
        channel_means[learner] = {}
        for j, channel in enumerate(channels):
            persistent = generator.normal(0.0, 2.0)
            values = []
            for occasion in (0, 1):
                value = truth[index] + common[index] + channel_bias[j] + persistent + generator.normal(0.0, transient_sd)
                value = max(0.0, float(value))
                values.append(round(value, 4))
                observations.append({"learner": learner, "occasion": occasion,
                                     "channel": channel, "minutes": round(value, 4)})
            channel_means[learner][channel] = float(np.mean(values))

    rows = [{"learner": learner, "household": households[index],
             "baseline_score": round(float(baseline[index]), 4),
             "score": round(float(score[index]), 4)} for index, learner in enumerate(identifiers)]

    audits = []
    if audit:
        for index, learner in enumerate(identifiers):
            if selection_depends_on_practice:
                high = truth[index] > 60.0
                keep = generator.random() < (0.75 if high else 0.35)
                probability = 0.75 if high else 0.35
            else:
                keep = generator.random() < 0.5
                probability = 0.5
            if not keep:
                continue
            manual_bias = generator.normal(0.0, 4.0)
            mid = truth[index] + manual_bias
            half = generator.uniform(8.0, 14.0)
            device_bias = generator.normal(0.0, 3.0)
            audits.append({
                "learner": learner,
                "selection_probability": probability,
                "device_minutes": round(float(truth[index] + device_bias), 4),
                "manual_interval": [round(float(max(0.0, mid - half)), 4),
                                    round(float(mid + half), 4)],
            })

    return _packet(rows, observations, audits), {
        "truth": truth,
        "common": common,
        "channel_means": channel_means,
        "true_effect_per_minute": true_effect_per_minute,
    }


class MeasurementErrorTests(unittest.TestCase):
    def setUp(self):
        self.packet, self.truth = _synthetic()
        self.result = analyze(self.packet, "dose_response", {"bootstrap_replicates": 120})["result"]

    def test_repeats_expose_transient_error_but_not_accuracy(self):
        per_channel = self.result["measurement_model"]["per_channel"]
        self.assertEqual(len(per_channel), 3)
        for entry in per_channel:
            self.assertGreater(entry["repeated_learners"], 0)
            self.assertGreater(entry["transient_variance"], 0.0)
            self.assertNotIn("accuracy", entry)

    def test_channel_agreement_is_not_a_reference_standard(self):
        decomposition = self.result["measurement_model"]["decomposition"]
        self.assertEqual(decomposition["identified_combination"], "Var(true minutes) + Var(common bias)")
        self.assertGreater(decomposition["shared_variance_S"], 0.0)
        self.assertLessEqual(decomposition["agreement_reliability_upper_bound"], 1.0)
        self.assertIsNone(self.result["association"]["chosen_reference"])

    def test_observed_association_is_attenuated_and_bound_is_below_truth(self):
        association = self.result["association"]
        true_per_60 = self.truth["true_effect_per_minute"] * 60.0
        observed = association["observed"]["per_60_minutes"]
        lower = association["identified_range"]["lower"]
        self.assertLess(observed, true_per_60)
        self.assertIsNotNone(lower)
        self.assertLess(lower, true_per_60)

    def test_scenario_grid_is_monotone_and_brackets_generating_slope(self):
        association = self.result["association"]
        values = [entry["per_60_minutes"] for entry in association["scenarios"]
                  if entry["per_60_minutes"] is not None]
        self.assertEqual(len(values), 4)
        self.assertEqual(values, sorted(values))
        true_per_60 = self.truth["true_effect_per_minute"] * 60.0
        # more assumed common bias can only move the corrected association away from
        # the attenuated observed value; the grid is not a confidence statement
        self.assertGreater(values[-1], values[0])
        self.assertGreater(values[-1], true_per_60 * 0.5)

    def test_households_share_reporter_error(self):
        diagnostic = self.result["diagnostics"]["household_transient_error"]
        self.assertEqual(diagnostic["pair_count"], 60)
        self.assertIsNotNone(diagnostic["correlation"])

    def test_result_is_json_finite(self):
        digest(self.result)


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.packet, _ = _synthetic(seed=11, learners=80, common_bias_sd=8.0)

    def test_same_seed_reproduces_and_interval_has_positive_width(self):
        first = analyze(self.packet, "dose_response", {"bootstrap_replicates": 120})
        second = analyze(self.packet, "dose_response", {"bootstrap_replicates": 120})
        self.assertEqual(first, second)
        interval = first["result"]["association"]["observed"]["interval_95"]
        self.assertLess(interval[0], interval[1])
        self.assertEqual(first["result"]["uncertainty"]["replicates_valid"], 120)

    def test_channels_are_ordered_and_permutation_invariant(self):
        shuffled = copy.deepcopy(self.packet)
        random.Random(3).shuffle(shuffled["observations"])
        random.Random(4).shuffle(shuffled["learners"])
        first = analyze(self.packet, "dose_response", {"bootstrap_replicates": 60})
        second = analyze(shuffled, "dose_response", {"bootstrap_replicates": 60})
        self.assertEqual(first["result"], second["result"])

    def test_unknown_and_out_of_range_options_are_rejected(self):
        for options in ({"device_is_truth": True},
                        {"common_bias_shares": [1.0]},
                        {"common_bias_shares": [-0.1]},
                        {"bootstrap_replicates": -3},
                        {"seed": 1.5}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    analyze(self.packet, "dose_response", options)


class SelectiveAuditTests(unittest.TestCase):
    def test_unweighted_audit_average_is_biased_and_weighted_is_not(self):
        packet, truth = _synthetic(seed=21, learners=200, common_bias_sd=6.0,
                                   selection_depends_on_practice=True)
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 40})["result"]
        means = result["diagnostics"]["audit_selection"]["mean_observed_minutes"]
        unweighted_gap = abs(means["audit_unweighted"] - means["cohort"])
        weighted_gap = abs(means["audit_inverse_probability_weighted"] - means["cohort"])
        self.assertGreater(unweighted_gap, 4.0)
        self.assertLess(weighted_gap, unweighted_gap)
        self.assertLess(weighted_gap, 3.0)
        selection = result["diagnostics"]["audit_selection"]
        self.assertEqual(selection["audited"], result["population"]["audited_learners"])

    def test_audits_are_anchors_with_stated_assumption_not_default(self):
        packet, _ = _synthetic(seed=22, learners=160)
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 40})["result"]
        anchors = result["association"]["audit_anchors"]
        self.assertTrue(anchors)
        for anchor in anchors:
            self.assertEqual(anchor["status"], "assumption_dependent")
            self.assertIn("instrument", anchor["assumption"])
            self.assertIn("cannot be verified", anchor["assumption"])
        self.assertIsNone(result["association"]["chosen_reference"])

    def test_reference_is_an_instrument_with_a_stated_assumption(self):
        packet, _ = _synthetic(seed=23, learners=160)
        request = {"bootstrap_replicates": 40, "reference": "audit_manual_mid"}
        result = analyze(packet, "dose_response", request)
        chosen = result["result"]["association"]["chosen_reference"]
        self.assertEqual(chosen["status"], "assumption_dependent")
        self.assertEqual(chosen["anchor"], "audit_manual_mid")
        self.assertIn("instrument", chosen["assumption"])
        self.assertIn("not assumed", chosen["assumption"])
        self.assertEqual(result["options"], request)
        with self.assertRaises(ValueError):
            analyze(packet, "dose_response", {"reference": "self_report"})


class BoundaryCaseTests(unittest.TestCase):
    def test_identical_exposure_is_not_identified(self):
        packet = local_packet("data/cases/boundary-1.json")
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 60})["result"]
        self.assertIn("no_exposure_variation", result["diagnostics"]["degenerate_conditions"])
        self.assertEqual(result["association"]["identified_range"]["status"], "not_identified")
        self.assertIsNone(result["association"]["observed"])
        self.assertEqual(result["support"]["supported_intervals"], [])
        self.assertFalse(result["support"]["exposure_variation"])

    def test_no_audits_keeps_partial_identification_without_anchor(self):
        packet = local_packet("data/cases/boundary-2.json")
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 60})["result"]
        self.assertIn("no_audits", result["diagnostics"]["degenerate_conditions"])
        self.assertIsNotNone(result["association"]["observed"])
        self.assertEqual(result["association"]["identified_range"]["status"], "partially_identified")
        self.assertIsNone(result["association"]["identified_range"]["upper"])
        self.assertTrue(all(anchor["per_60_minutes"] is None
                            for anchor in result["association"]["audit_anchors"]))
        self.assertIsNone(result["association"]["chosen_reference"])

    def test_household_shared_offsets_are_not_resolved_by_agreement(self):
        packet = local_packet("data/cases/boundary-3.json")
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 60})["result"]
        decomposition = result["measurement_model"]["decomposition"]
        self.assertEqual(decomposition["agreement_reliability_upper_bound"], 1.0)
        self.assertIn("no_within_household_contrast", result["diagnostics"]["degenerate_conditions"])
        check = result["diagnostics"]["audit_consistency"]["reliability_check"]
        self.assertLess(check["audit_implied_composite_reliability"], 0.5)
        self.assertIsNone(result["association"]["identified_range"]["upper"])

    def test_single_occasion_still_bounds_but_cannot_estimate_repeats(self):
        packet, _ = _synthetic(seed=31, learners=60)
        packet["observations"] = [row for row in packet["observations"] if row["occasion"] == 0]
        packet["revision"] = "single-occasion"
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 40})["result"]
        self.assertIn("no_repeated_measurements", result["diagnostics"]["degenerate_conditions"])
        for entry in result["measurement_model"]["per_channel"]:
            self.assertIsNone(entry["transient_variance"])
            self.assertEqual(entry["repeated_learners"], 0)
        # agreement across channels still identifies the shared variance, so a
        # one-sided bound remains; no full correction is possible
        self.assertEqual(result["association"]["identified_range"]["status"], "partially_identified")
        self.assertIsNone(result["association"]["identified_range"]["upper"])


class NonCausalTests(unittest.TestCase):
    def test_interpretation_separates_association_from_causal_gain(self):
        packet, _ = _synthetic(seed=41, learners=120)
        result = analyze(packet, "dose_response", {"bootstrap_replicates": 60})["result"]
        self.assertEqual(result["association"]["causal_status"],
                         result["estimand"]["causal_status"])
        self.assertIn("not a causal", result["association"]["causal_status"])
        self.assertTrue(result["interpretation"]["not_a_causal_effect"])
        self.assertIn("does not pick a correction factor",
                      result["interpretation"]["default_resolution"])
        self.assertIn("not a causal gain",
                      result["association"]["within_household"]["interpretation"])


if __name__ == "__main__":
    unittest.main()
