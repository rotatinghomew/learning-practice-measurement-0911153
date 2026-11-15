import argparse
import json
import tempfile
import threading
from urllib.request import Request, urlopen

from app.catalog import local_packet
from app.domain import ANALYSES
from app.server import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default=next(iter(ANALYSES)))
    parser.add_argument("--packet", default="data/current.json")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        catalog = make_server(0, "catalog", "", arguments.packet, directory)
        source_url = f"http://127.0.0.1:{catalog.server_port}"
        analysis = make_server(0, "analysis", source_url, arguments.packet, directory)
        threads = [threading.Thread(target=server.serve_forever, daemon=True)
                   for server in (catalog, analysis)]
        for thread in threads:
            thread.start()
        try:
            address = f"http://127.0.0.1:{analysis.server_port}"
            body = json.dumps({"revision": local_packet(arguments.packet)["revision"],
                               "analysis": arguments.analysis, "options": {}}).encode()
            with urlopen(Request(address + "/analyses", data=body,
                                 headers={"Content-Type": "application/json"}), timeout=60) as response:
                created = json.load(response)
            with urlopen(address + "/results/" + created["id"], timeout=10) as response:
                saved = json.load(response)
            if saved != created["envelope"]:
                raise ValueError("saved result differs")
            print(json.dumps({"artifact_id": created["id"], "saved_result": saved},
                             ensure_ascii=False, indent=2, allow_nan=False))
        finally:
            for server in (analysis, catalog):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=5)


if __name__ == "__main__":
    main()
