import argparse
import threading

from app.server import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--packet", default="data/current.json")
    parser.add_argument("--results", default="var/results")
    arguments = parser.parse_args()
    catalog = make_server(0, "catalog", "", arguments.packet, arguments.results)
    source_url = f"http://127.0.0.1:{catalog.server_port}"
    analysis = make_server(arguments.port, "analysis", source_url, arguments.packet,
                           arguments.results, arguments.host)
    source_thread = threading.Thread(target=catalog.serve_forever, daemon=True)
    source_thread.start()
    try:
        analysis.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        analysis.server_close()
        catalog.shutdown()
        catalog.server_close()
        source_thread.join(timeout=5)


if __name__ == "__main__":
    main()
