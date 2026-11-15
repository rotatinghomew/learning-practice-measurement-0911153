import unittest
from app.catalog import local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def result(self):
        return analyze(self.packet, next(iter(ANALYSES)))["result"]

    def test_repeatability_does_not_name_a_true_channel(self):
        result = self.result()
        self.assertEqual(len(result["channels"]), 3)
        self.assertTrue(all(row["repeat_pairs"] > 0 for row in result["channels"]))

    def test_negative_minutes_rejected(self):
        self.packet["observations"][0]["minutes"] = -1
        with self.assertRaises(ValueError):
            validate(self.packet)
