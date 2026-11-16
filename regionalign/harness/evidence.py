"""Small JSON evidence helpers shared by RegionAlign experiments."""
from . import core as P


def sanitize(value):
    """JSON-safe copy: non-finite floats become null, with verdicts left unchanged."""
    if isinstance(value, float):
        return P.reportable(value)
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value
