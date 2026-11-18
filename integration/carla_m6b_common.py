"""Shared, behavior-neutral primitives used by the M6B integration modules.

The helpers in this module deliberately contain no CARLA imports and no
milestone-specific policy.  Keeping them here prevents small copies of the
same numeric, angle, and provenance logic from drifting apart while allowing
the existing modules to re-export the original helper names.
"""
from __future__ import annotations

# ``hashlib`` supplies the deterministic provenance digest used in generated
# JSON reports; it is intentionally based on file bytes rather than parsed data.
import hashlib
# ``math`` provides finite-number checks and the trigonometric angle wrapper.
import math
# ``Path`` gives the digest helper a consistent filesystem interface.
from pathlib import Path

# Every M6B sample is one fixed simulation step apart.
DT = 0.05


def finite(name, value):
    """Return ``value`` as a finite float or raise a descriptive ``ValueError``."""
    # Conversion accepts numeric strings read from CSV files as well as numbers
    # produced directly by CARLA or by the offline fixtures.
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not numeric') from exc
    # NaN and infinities would poison geometry, control, and report calculations.
    if not math.isfinite(number):
        raise ValueError(f'{name} is not finite')
    return number


def wrap(angle):
    """Normalize an angle in radians to the principal ``[-pi, pi]`` interval."""
    # atan2(sin, cos) performs wrapping without a discontinuous modulo branch.
    return math.atan2(math.sin(angle), math.cos(angle))


def digest(path):
    """Return the SHA-256 digest of a file's exact bytes."""
    # Hashing bytes preserves provenance even when equivalent JSON parses
    # differently because of formatting or key ordering.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
