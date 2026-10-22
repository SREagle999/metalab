"""Small dependency-free local peak extraction and continuity tracking."""

from dataclasses import dataclass

import numpy as np

from ..errors import MaterialError


@dataclass(frozen=True)
class Peak:
    x: float
    height: float
    width: float
    index: int


def find_peaks_1d(x, values, *, prominence=0.0):
    """Find strict local maxima and estimate half-prominence widths by interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise MaterialError("x and values must be finite one-dimensional arrays of equal length (at least 3).")
    if np.any(np.diff(x) <= 0) or not np.isfinite(prominence) or prominence < 0:
        raise MaterialError("x must be strictly increasing and prominence must be finite and nonnegative.")
    peaks = []
    for i in range(1, x.size - 1):
        if not (y[i] > y[i - 1] and y[i] >= y[i + 1]):
            continue
        left_min = np.min(y[:i + 1])
        right_min = np.min(y[i:])
        prom = y[i] - max(left_min, right_min)
        if prom < prominence:
            continue
        half = y[i] - prom / 2
        left = i
        while left > 0 and y[left] > half:
            left -= 1
        right = i
        while right < y.size - 1 and y[right] > half:
            right += 1
        xleft = _crossing(x[left], y[left], x[min(left + 1, i)], y[min(left + 1, i)], half)
        xright = _crossing(x[max(i, right - 1)], y[max(i, right - 1)], x[right], y[right], half)
        peaks.append(Peak(float(x[i]), float(y[i]), float(max(0.0, xright - xleft)), i))
    return peaks


def _crossing(x0, y0, x1, y1, level):
    if y1 == y0:
        return float((x0 + x1) / 2)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def track_peak_branches(x, spectra, *, prominence=0.0, max_jump=None):
    """Greedily match peaks on adjacent spectra by minimum x displacement.

    Missing branch samples are ``None``. Returns one list per branch.
    """
    x = np.asarray(x, dtype=float)
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[1] != x.size:
        raise MaterialError("spectra must have shape (n_scan, len(x)).")
    if x.ndim != 1 or np.any(np.diff(x) <= 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(spectra)):
        raise MaterialError("x and spectra must be finite; x must be strictly increasing.")
    if max_jump is None:
        max_jump = float(x[-1] - x[0])
    if not np.isfinite(max_jump) or max_jump < 0:
        raise MaterialError("max_jump must be finite and nonnegative.")
    branches = []
    for row_index, row in enumerate(spectra):
        peaks = find_peaks_1d(x, row, prominence=prominence)
        if row_index == 0:
            branches = [[peak] for peak in peaks]
            continue
        candidates = []
        for bi, branch in enumerate(branches):
            previous = next((item for item in reversed(branch) if item is not None), None)
            if previous is not None:
                for pi, peak in enumerate(peaks):
                    candidates.append((abs(peak.x - previous.x), bi, pi))
        assigned_branches, assigned_peaks = set(), set()
        links = {}
        for distance, bi, pi in sorted(candidates):
            if distance <= max_jump and bi not in assigned_branches and pi not in assigned_peaks:
                links[bi] = peaks[pi]
                assigned_branches.add(bi)
                assigned_peaks.add(pi)
        for bi, branch in enumerate(branches):
            branch.append(links.get(bi))
        for pi, peak in enumerate(peaks):
            if pi not in assigned_peaks:
                branches.append([None] * row_index + [peak])
    return branches
