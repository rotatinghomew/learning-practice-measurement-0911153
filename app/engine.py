from app.domain import ANALYSES, validate
from app.protocol import digest


def analyze(packet, analysis, options=None):
    if analysis not in ANALYSES:
        raise ValueError(f"unknown analysis: {analysis}")
    validate(packet)
    configuration = {} if options is None else options
    if not isinstance(configuration, dict):
        raise ValueError("options must be an object")
    result = ANALYSES[analysis](packet, configuration)
    digest(result)
    return {
        "analysis": analysis,
        "analysis_version": 1,
        "revision": packet["revision"],
        "source_sha256": digest(packet),
        "options": configuration,
        "result": result,
    }
