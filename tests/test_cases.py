import json
import unittest
from pathlib import Path
from app.catalog import local_packet
from app.domain import ANALYSES
from app.engine import analyze


class BoundaryDataTests(unittest.TestCase):
    def test_boundary_packets_preserve_current_product_contract(self):
        manifest = json.loads(Path("data/cases/index.json").read_text())
        self.assertEqual(len(manifest), 3)
        for row in manifest:
            with self.subTest(packet=row["path"]):
                packet = local_packet(row["path"])
                result = analyze(packet, next(iter(ANALYSES)))
                self.assertEqual(result["revision"], packet["revision"])
                self.assertTrue(result["result"])
