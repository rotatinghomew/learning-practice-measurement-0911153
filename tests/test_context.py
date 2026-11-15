import unittest
from app.catalog import local_packet
from app.measurements import measurement_profile as profile


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_agreement_is_not_declared_accuracy(self):
        self.assertIsNone(profile(self.packet)["reference_channel"])
        self.assertEqual(len(profile(self.packet)["channel_agreement"]), 3)

    def test_validation_selection_varies(self):
        minimum, maximum = profile(self.packet)["audit_probability_range"]
        self.assertLess(minimum, maximum)
