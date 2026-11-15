import copy
import unittest
from app.catalog import local_packet
from app.domain import ANALYSES
from app.engine import analyze
from app.protocol import digest


class FeatureInterfaceAcceptance(unittest.TestCase):
    def test_requested_analysis_is_registered(self):
        self.assertIn("dose_response", ANALYSES)

    def test_requested_analysis_keeps_provenance_without_mutating_source(self):
        packet = local_packet("data/current.json")
        before = copy.deepcopy(packet)
        result = analyze(packet, "dose_response")
        self.assertEqual(packet, before)
        self.assertEqual(result["source_sha256"], digest(before))
        self.assertEqual(result["analysis"], "dose_response")
        self.assertIsInstance(result["result"], dict)
        self.assertTrue(result["result"])
