import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from app.catalog import fetch_packet, local_packet
from app.engine import analyze
from app.protocol import digest, encode
from app.store import ResultStore


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, mode, catalog, packet_path, store, **kwargs):
        self.mode = mode
        self.catalog = catalog
        self.packet_path = packet_path
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *arguments):
        return

    def reply(self, status, value):
        content = encode(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        route = urlsplit(self.path).path
        try:
            if route == "/health":
                self.reply(200, {"status": "ok", "mode": self.mode})
            elif self.mode == "catalog" and route.startswith("/packets/"):
                packet = local_packet(self.packet_path)
                if route.rsplit("/", 1)[1] != packet["revision"]:
                    raise FileNotFoundError("unknown revision")
                self.reply(200, {"packet": packet, "sha256": digest(packet)})
            elif self.mode == "analysis" and route.startswith("/results/"):
                self.reply(200, self.store.get(route.rsplit("/", 1)[1]))
            else:
                self.reply(404, {"error": "unknown route"})
        except FileNotFoundError:
            self.reply(404, {"error": "not found"})
        except ValueError as error:
            self.reply(422, {"error": str(error)})

    def do_POST(self):
        if self.mode != "analysis" or self.path != "/analyses":
            self.reply(404, {"error": "unknown route"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1024 * 1024:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            packet, source_sha = fetch_packet(self.catalog, request["revision"])
            envelope = analyze(packet, request["analysis"], request.get("options"))
            if envelope["source_sha256"] != source_sha:
                raise ValueError("source changed")
            identifier = self.store.put(envelope)
            self.reply(201, {"id": identifier, "envelope": envelope})
        except (ValueError, KeyError, TypeError) as error:
            self.reply(422, {"error": str(error)})
        except OSError as error:
            self.reply(503, {"error": str(error)})


def make_server(port, mode, catalog, packet_path, result_directory, host="127.0.0.1"):
    handler = partial(Handler, mode=mode, catalog=catalog, packet_path=packet_path,
                      store=ResultStore(result_directory))
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["catalog", "analysis"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--catalog", default="http://127.0.0.1:8311")
    parser.add_argument("--packet", default="data/current.json")
    parser.add_argument("--results", default="var/results")
    arguments = parser.parse_args()
    server = make_server(arguments.port, arguments.mode, arguments.catalog,
                         arguments.packet, arguments.results)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
