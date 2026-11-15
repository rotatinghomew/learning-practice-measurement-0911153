import hashlib
import json
import math


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected finite number")
    return value


def unique(rows, field):
    identifiers = [row[field] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate {field}")


def table_mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
