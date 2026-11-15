import json
from pathlib import Path
from urllib.request import urlopen

from app.protocol import digest


def local_packet(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_packet(base_url, revision):
    with urlopen(f"{base_url}/packets/{revision}", timeout=10) as response:
        envelope = json.load(response)
    packet = envelope["packet"]
    if packet["revision"] != revision or digest(packet) != envelope["sha256"]:
        raise ValueError("catalog revision or checksum mismatch")
    return packet, envelope["sha256"]
