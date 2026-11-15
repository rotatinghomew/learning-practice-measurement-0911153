import json
import os
import tempfile
from pathlib import Path

from app.protocol import digest, encode


class ResultStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, envelope):
        identifier = digest(envelope)
        target = self.root / f"{identifier}.json"
        descriptor, pending = tempfile.mkstemp(dir=self.root, prefix=".pending-")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encode(envelope))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(pending, target)
            except FileExistsError:
                if target.read_bytes() != encode(envelope):
                    raise ValueError("artifact content conflict")
        finally:
            os.unlink(pending)
        return identifier

    def get(self, identifier):
        if len(identifier) != 64 or any(character not in "0123456789abcdef" for character in identifier):
            raise ValueError("invalid artifact id")
        value = json.loads((self.root / f"{identifier}.json").read_text(encoding="utf-8"))
        if digest(value) != identifier:
            raise ValueError("artifact checksum mismatch")
        return value
