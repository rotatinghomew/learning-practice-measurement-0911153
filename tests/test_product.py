import copy
import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.catalog import fetch_packet, local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze
from app.protocol import digest
from app.server import make_server
from app.store import ResultStore


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")
        self.analysis = next(iter(ANALYSES))

    def test_baseline_is_deterministic(self):
        self.assertEqual(analyze(self.packet, self.analysis), analyze(self.packet, self.analysis))

    def test_missing_revision_is_rejected(self):
        packet = copy.deepcopy(self.packet)
        del packet["revision"]
        with self.assertRaises((ValueError, KeyError)):
            validate(packet)

    def test_unknown_analysis_is_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            analyze(self.packet, "unknown-analysis")

    def test_source_and_options_are_fingerprinted(self):
        first = analyze(self.packet, self.analysis, {"note": "first"})
        second = analyze(self.packet, self.analysis, {"note": "second"})
        self.assertNotEqual(digest(first), digest(second))
        self.assertEqual(first["source_sha256"], digest(self.packet))

    def test_json_rejects_nonfinite_results(self):
        with self.assertRaises(ValueError):
            digest({"value": float("nan")})

    def test_store_has_immutable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory)
            first = analyze(self.packet, self.analysis)
            identifier = store.put(first)
            self.assertEqual(store.put(first), identifier)
            self.assertEqual(store.get(identifier), first)
            with self.assertRaises(ValueError):
                store.get("../../data/current")

    def test_real_http_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = make_server(0, "catalog", "", "data/current.json", directory)
            address = f"http://127.0.0.1:{catalog.server_port}"
            server = make_server(0, "analysis", address, "data/current.json", directory)
            threads = [threading.Thread(target=item.serve_forever, daemon=True) for item in (catalog, server)]
            for thread in threads:
                thread.start()
            try:
                packet, checksum = fetch_packet(address, self.packet["revision"])
                self.assertEqual(checksum, digest(packet))
                request = Request(f"http://127.0.0.1:{server.server_port}/analyses",
                                  data=json.dumps({"revision": self.packet["revision"],
                                                   "analysis": self.analysis}).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=10) as response:
                    created = json.load(response)
                    self.assertEqual(response.status, 201)
                with urlopen(f"http://127.0.0.1:{server.server_port}/results/{created['id']}", timeout=10) as response:
                    self.assertEqual(json.load(response), created["envelope"])
                request.data = b'{"revision":"missing","analysis":"unknown-analysis"}'
                with self.assertRaises(HTTPError) as failure:
                    urlopen(request, timeout=10)
                self.assertEqual(failure.exception.code, 503)
            finally:
                for item in (server, catalog):
                    item.shutdown()
                    item.server_close()
                for thread in threads:
                    thread.join(timeout=5)
