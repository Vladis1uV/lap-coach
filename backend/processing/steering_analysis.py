"""
====================
Steering-wheel–specific corner detection and lap comparison.

Steering is a *continuous waveform* — the atomic unit is a **corner**, defined
by three phases:

  ┌─────────────────────────────────────────────────────────┐
  │  TURN-IN  │       APEX        │       UNWIND            │
  │  (rising  │  (peak |steering| │  (falling edge back     │
  │   edge)   │   held briefly)   │   toward zero)          │
  └─────────────────────────────────────────────────────────┘

For each reference corner we extract:
  - turn_in_arc   : where |steering| crosses the entry threshold going up
  - apex_arc      : where |steering| is maximum
  - apex_angle    : peak steering angle (rad)
  - unwind_arc    : where |steering| crosses the exit threshold going down
  - direction     : +1 (left) / -1 (right)

We then match each reference corner to the nearest slow-lap corner and compare
all three phases independently, producing verdicts:

  TURN_IN  : on time / too late / too early
  APEX     : similar / too much angle / too little angle
  UNWIND   : on time / too early (rushed unwind) / too late (held lock)

We also run a **waveform cross-correlation** inside each matched corner window
to get a continuous lag estimate — useful when the overall shape is right but
slightly shifted.

Public API
----------
    detect_corners(states, arc, **kwargs) -> list[Corner]
    match_corners(ref, slow, ...)         -> list[CornerMatch]
    print_steering_recommendations(matches)
    plot_steering_analysis(...)

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, find_peaks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(signal: np.ndarray, window: int = 9) -> np.ndarray:
    return uniform_filter1d(signal.astype(float), size=max(1, window))


def _arc_to_samples(arc: np.ndarray, metres: float) -> int:
    if len(arc) < 2 or arc[-1] <= arc[0]:
        return 1
    avg = (arc[-1] - arc[0]) / (len(arc) - 1)
    return max(1, int(metres / avg))


def _interp_arc(arc: np.ndarray, idx: int) -> float:
    return float(arc[min(idx, len(arc) - 1)])


# ---------------------------------------------------------------------------
# Corner dataclass
# ---------------------------------------------------------------------------

@dataclass
class Corner:
    """
    A single steering corner event, defined by its three phases.

    All arc positions are in metres along the track.
    Angles are in radians (positive = left turn in vehicle convention).
    """
    # Indices into the states list
    turn_in_idx:  int
    apex_idx:     int
    unwind_idx:   int

    # Arc-length positions (m)
    turn_in_arc:  float
    apex_arc:     float
    unwind_arc:   float

    # Characterisation
    apex_angle:   float   # signed peak steering angle (rad)
    direction:    int     # +1 = left, -1 = right
    entry_rate:   float   # rad/m — how quickly the driver winds on lock
    exit_rate:    float   # rad/m — how quickly they unwind

    # Position on track (for map overlay)
    x_turn_in:   float
    y_turn_in:   float
    x_apex:      float
    y_apex:      float


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

def detect_corners(
    states: list,
    arc: np.ndarray,
    *,
    entry_threshold: float  = 0.04,   # |steering| to mark turn-in start (rad)
    exit_threshold: float   = 0.04,   # |steering| to mark unwind end (rad)
    min_apex_angle: float   = 0.08,   # minimum peak |steering| to count as corner (rad)
    min_gap_m: float        = 30.0,   # minimum metres between two corners
    smooth_window: int      = 11,     # pre-smoothing window
    walk_limit_m: float     = 60.0,   # max distance to walk back/forward from apex
) -> list[Corner]:
    """
    Detect corners in a steering trace.

    Algorithm
    ---------
    1. Smooth the raw steering signal.
    2. Take |steering| and find peaks with scipy.find_peaks (prominence-based).
    3. For each peak (= apex):
       a. Walk *backward* from the apex until |steering| drops below
          entry_threshold → turn-in point.
       b. Walk *forward* from the apex until |steering| drops below
          exit_threshold → unwind point.
       c. Compute entry/exit rates (rad/m).
    4. Enforce minimum gap between consecutive corners.
    """
    raw   = np.array([s.steering for s in states])
    steer = _smooth(raw, smooth_window)
    abst  = np.abs(steer)

    # Minimum gap in samples
    min_gap_s = _arc_to_samples(arc, min_gap_m)
    walk_lim  = _arc_to_samples(arc, walk_limit_m)

    # Find apex candidates
    apex_indices, _ = find_peaks(
        abst,
        height=min_apex_angle,
        prominence=min_apex_angle * 0.5,
        distance=min_gap_s,
    )

    corners: list[Corner] = []

    for apex_idx in apex_indices:
        apex_angle = float(steer[apex_idx])
        direction  = 1 if apex_angle >= 0 else -1

        # --- Walk backward to find turn-in ---
        turn_in_idx = apex_idx
        for k in range(apex_idx - 1, max(0, apex_idx - walk_lim) - 1, -1):
            if abst[k] < entry_threshold:
                turn_in_idx = k
                break
        else:
            turn_in_idx = max(0, apex_idx - walk_lim)

        # --- Walk forward to find unwind ---
        unwind_idx = apex_idx
        for k in range(apex_idx + 1, min(len(steer), apex_idx + walk_lim)):
            if abst[k] < exit_threshold:
                unwind_idx = k
                break
        else:
            unwind_idx = min(len(steer) - 1, apex_idx + walk_lim)

        # --- Compute rates ---
        entry_dist = arc[apex_idx] - arc[turn_in_idx]
        exit_dist  = arc[unwind_idx] - arc[apex_idx]
        entry_rate = abs(apex_angle) / entry_dist if entry_dist > 0.1 else 0.0
        exit_rate  = abs(apex_angle) / exit_dist  if exit_dist  > 0.1 else 0.0

        corners.append(Corner(
            turn_in_idx  = int(turn_in_idx),
            apex_idx     = int(apex_idx),
            unwind_idx   = int(unwind_idx),
            turn_in_arc  = _interp_arc(arc, turn_in_idx),
            apex_arc     = _interp_arc(arc, apex_idx),
            unwind_arc   = _interp_arc(arc, unwind_idx),
            apex_angle   = apex_angle,
            direction    = direction,
            entry_rate   = entry_rate,
            exit_rate    = exit_rate,
            x_turn_in    = states[turn_in_idx].x,
            y_turn_in    = states[turn_in_idx].y,
            x_apex       = states[apex_idx].x,
            y_apex       = states[apex_idx].y,
        ))

    return corners


# ---------------------------------------------------------------------------
# Waveform cross-correlation inside a corner window
# ---------------------------------------------------------------------------

def _corner_lag_m(
    ref_corner: Corner,
    ref_arc: np.ndarray,
    ref_steer: np.ndarray,
    slow_arc: np.ndarray,
    slow_steer: np.ndarray,
    *,
    context_m: float = 20.0,    # metres of context around the corner window
) -> float:
    """
    Estimate the arc-length lag between the slow and reference steering waveforms
    inside the corner window using normalised cross-correlation.

    Returns lag in metres (positive = slow lap is shifted later on track).
    A common arc grid is built by resampling both signals at 1 m resolution
    to make the lag physically meaningful.
    """
    lo = ref_corner.turn_in_arc - context_m
    hi = ref_corner.unwind_arc  + context_m

    # 1 m resolution common grid
    grid = np.arange(lo, hi, 1.0)
    if len(grid) < 5:
        return 0.0

    ref_on_grid  = np.interp(grid, ref_arc,  ref_steer)
    slow_on_grid = np.interp(grid, slow_arc, slow_steer)

    # Normalised cross-correlation
    a = ref_on_grid  - ref_on_grid.mean()
    b = slow_on_grid - slow_on_grid.mean()
    norm = (np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0

    xcorr = correlate(b, a, mode="full") / norm
    lags  = np.arange(-(len(a) - 1), len(b))   # in samples = metres on 1 m grid
    lag_m = float(lags[np.argmax(xcorr)])       # positive = slow is later
    return lag_m


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class TurnInVerdict(Enum):
    ON_TIME   = auto()
    TOO_LATE  = auto()   # slow driver turns in later (further on track)
    TOO_EARLY = auto()   # slow driver turns in earlier


class ApexVerdict(Enum):
    SIMILAR      = auto()
    TOO_MUCH     = auto()   # slow driver uses more lock (over-rotation)
    TOO_LITTLE   = auto()   # slow driver doesn't reach the same angle


class UnwindVerdict(Enum):
    ON_TIME      = auto()
    TOO_EARLY    = auto()   # slow driver starts unwinding before apex work is done
    TOO_LATE     = auto()   # slow driver holds lock longer (tight exit)


# ---------------------------------------------------------------------------
# CornerMatch
# ---------------------------------------------------------------------------

@dataclass
class CornerMatch:
    ref:  Corner
    slow: Corner | None

    # Per-phase verdicts
    turn_in_verdict: TurnInVerdict
    apex_verdict:    ApexVerdict
    unwind_verdict:  UnwindVerdict

    # Offsets in metres / radians
    turn_in_offset_m:  float   # slow.turn_in_arc  − ref.turn_in_arc
    apex_offset_m:     float   # slow.apex_arc      − ref.apex_arc
    unwind_offset_m:   float   # slow.unwind_arc    − ref.unwind_arc
    apex_angle_delta:  float   # slow.apex_angle    − ref.apex_angle  (rad)
    xcorr_lag_m:       float   # cross-correlation lag (m)

    recommendations: list[str]  # human-readable coaching lines


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _nearest_corner(
    target_arc: float,
    candidates: list[Corner],
    radius_m: float,
) -> Corner | None:
    within = [c for c in candidates if abs(c.apex_arc - target_arc) <= radius_m]
    return min(within, key=lambda c: abs(c.apex_arc - target_arc)) if within else None


def match_corners(
    ref_corners:  list[Corner],
    slow_corners: list[Corner],
    ref_arc:      np.ndarray,
    slow_arc:     np.ndarray,
    ref_steer:    np.ndarray,
    slow_steer:   np.ndarray,
    *,
    on_time_window_m:  float = 8.0,    # ±m considered "on time" for turn-in / unwind
    search_radius_m:   float = 60.0,
    apex_angle_tol:    float = 0.05,   # rad tolerance for "similar" apex
) -> list[CornerMatch]:
    """
    For each reference corner, find the nearest slow-lap corner and compare
    all three phases independently.
    """
    results: list[CornerMatch] = []

    for ref in ref_corners:
        slow = _nearest_corner(ref.apex_arc, slow_corners, search_radius_m)

        if slow is None:
            results.append(CornerMatch(
                ref=ref, slow=None,
                turn_in_verdict=TurnInVerdict.TOO_LATE,
                apex_verdict=ApexVerdict.TOO_LITTLE,
                unwind_verdict=UnwindVerdict.ON_TIME,
                turn_in_offset_m=float("nan"),
                apex_offset_m=float("nan"),
                unwind_offset_m=float("nan"),
                apex_angle_delta=float("nan"),
                xcorr_lag_m=float("nan"),
                recommendations=[
                    f"At ~{ref.apex_arc:.0f} m no matching corner found in the slow lap. "
                    f"The reference takes a {'left' if ref.direction > 0 else 'right'} "
                    f"corner (peak {abs(ref.apex_angle):.3f} rad) that appears to be "
                    f"missing from your lap entirely."
                ],
            ))
            continue

        # --- Phase offsets ---
        ti_off  = slow.turn_in_arc  - ref.turn_in_arc
        ap_off  = slow.apex_arc     - ref.apex_arc
        uw_off  = slow.unwind_arc   - ref.unwind_arc
        ang_d   = abs(slow.apex_angle) - abs(ref.apex_angle)

        # --- Cross-correlation lag ---
        xcorr_lag = _corner_lag_m(
            ref, ref_arc, ref_steer, slow_arc, slow_steer
        )

        # --- Classify turn-in ---
        if abs(ti_off) <= on_time_window_m:
            ti_verdict = TurnInVerdict.ON_TIME
        elif ti_off > 0:
            ti_verdict = TurnInVerdict.TOO_LATE
        else:
            ti_verdict = TurnInVerdict.TOO_EARLY

        # --- Classify apex angle ---
        if abs(ang_d) <= apex_angle_tol:
            ap_verdict = ApexVerdict.SIMILAR
        elif ang_d > 0:
            ap_verdict = ApexVerdict.TOO_MUCH
        else:
            ap_verdict = ApexVerdict.TOO_LITTLE

        # --- Classify unwind ---
        if abs(uw_off) <= on_time_window_m:
            uw_verdict = UnwindVerdict.ON_TIME
        elif uw_off < 0:
            uw_verdict = UnwindVerdict.TOO_EARLY
        else:
            uw_verdict = UnwindVerdict.TOO_LATE

        # --- Build recommendations ---
        recs: list[str] = []
        direction_str = "left" if ref.direction > 0 else "right"
        corner_id     = f"~{ref.apex_arc:.0f} m ({direction_str})"

        if ti_verdict == TurnInVerdict.TOO_LATE:
            recs.append(
                f"[{corner_id}] Turn in {abs(ti_off):.1f} m earlier — "
                f"you turned at {slow.turn_in_arc:.0f} m, reference at {ref.turn_in_arc:.0f} m. "
                f"Late turn-in forces a tighter radius and kills exit speed."
            )
        elif ti_verdict == TurnInVerdict.TOO_EARLY:
            recs.append(
                f"[{corner_id}] You're turning in {abs(ti_off):.1f} m too early "
                f"({slow.turn_in_arc:.0f} m vs {ref.turn_in_arc:.0f} m). "
                f"Early turn-in leads to a tighter apex and compromises the exit."
            )

        if ap_verdict == ApexVerdict.TOO_LITTLE:
            recs.append(
                f"[{corner_id}] Your peak lock ({abs(slow.apex_angle):.3f} rad) is "
                f"{abs(ang_d):.3f} rad less than the reference ({abs(ref.apex_angle):.3f} rad). "
                f"You may be running a wider arc than necessary — commit to the apex."
            )
        elif ap_verdict == ApexVerdict.TOO_MUCH:
            recs.append(
                f"[{corner_id}] Your peak lock ({abs(slow.apex_angle):.3f} rad) exceeds "
                f"the reference by {abs(ang_d):.3f} rad — you're over-rotating. "
                f"This usually indicates a late turn-in forcing a tighter line."
            )

        if uw_verdict == UnwindVerdict.TOO_EARLY:
            recs.append(
                f"[{corner_id}] You start unwinding {abs(uw_off):.1f} m before the reference. "
                f"Unwinding early trades rotation for stability — "
                f"consider holding a fraction more lock through the apex."
            )
        elif uw_verdict == UnwindVerdict.TOO_LATE:
            recs.append(
                f"[{corner_id}] You're holding lock {abs(uw_off):.1f} m longer than the reference. "
                f"A late unwind tightens the exit line — try to feed out the steering sooner."
            )

        # Cross-correlation summary (only if meaningfully different from phase offsets)
        if abs(xcorr_lag) > on_time_window_m and abs(xcorr_lag - ti_off) > 5.0:
            recs.append(
                f"[{corner_id}] Waveform correlation suggests your entire steering "
                f"input is shifted {xcorr_lag:+.1f} m relative to the reference "
                f"(positive = you act later on track)."
            )

        # Entry / exit rate comparison
        if ref.entry_rate > 0 and slow.entry_rate > 0:
            rate_ratio = slow.entry_rate / ref.entry_rate
            if rate_ratio < 0.7:
                recs.append(
                    f"[{corner_id}] You're winding on lock more slowly than the reference "
                    f"({slow.entry_rate:.3f} vs {ref.entry_rate:.3f} rad/m). "
                    f"A hesitant turn-in can indicate a lack of confidence in front grip."
                )
            elif rate_ratio > 1.4:
                recs.append(
                    f"[{corner_id}] You're winding on lock more aggressively "
                    f"({slow.entry_rate:.3f} vs {ref.entry_rate:.3f} rad/m). "
                    f"A very sharp turn-in can unsettle the rear."
                )

        if not recs:
            recs.append(
                f"[{corner_id}] Steering input matches the reference well "
                f"(turn-in Δ{ti_off:+.1f} m, apex Δ{ang_d:+.3f} rad, "
                f"unwind Δ{uw_off:+.1f} m)."
            )

        results.append(CornerMatch(
            ref=ref, slow=slow,
            turn_in_verdict=ti_verdict,
            apex_verdict=ap_verdict,
            unwind_verdict=uw_verdict,
            turn_in_offset_m=ti_off,
            apex_offset_m=ap_off,
            unwind_offset_m=uw_off,
            apex_angle_delta=ang_d,
            xcorr_lag_m=xcorr_lag,
            recommendations=recs,
        ))

    return results


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_ICONS = {
    TurnInVerdict.ON_TIME:       "✅",
    TurnInVerdict.TOO_LATE:      "⚠️ ",
    TurnInVerdict.TOO_EARLY:     "⚠️ ",
    ApexVerdict.SIMILAR:         "✅",
    ApexVerdict.TOO_MUCH:        "⚠️ ",
    ApexVerdict.TOO_LITTLE:      "⚠️ ",
    UnwindVerdict.ON_TIME:       "✅",
    UnwindVerdict.TOO_EARLY:     "⚠️ ",
    UnwindVerdict.TOO_LATE:      "⚠️ ",
}


def print_steering_recommendations(matches: list[CornerMatch]) -> None:
    print("\n" + "=" * 64)
    print(f"  STEERING — {len(matches)} reference corners analysed")
    print("=" * 64)

    for m in matches:
        ti_icon = _ICONS.get(m.turn_in_verdict, "?")
        ap_icon = _ICONS.get(m.apex_verdict,    "?")
        uw_icon = _ICONS.get(m.unwind_verdict,  "?")

        ref_dir = "←" if m.ref.direction > 0 else "→"
        print(f"\n  Corner {ref_dir}  apex ~{m.ref.apex_arc:.0f} m  "
              f"|  turn-in {ti_icon}  apex-angle {ap_icon}  unwind {uw_icon}")
        for rec in m.recommendations:
            print(f"    • {rec}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_TI_COLORS = {
    TurnInVerdict.ON_TIME:   "#2ecc71",
    TurnInVerdict.TOO_LATE:  "#e74c3c",
    TurnInVerdict.TOO_EARLY: "#f1c40f",
}
_AP_COLORS = {
    ApexVerdict.SIMILAR:    "#2ecc71",
    ApexVerdict.TOO_MUCH:   "#e74c3c",
    ApexVerdict.TOO_LITTLE: "#f1c40f",
}
_UW_COLORS = {
    UnwindVerdict.ON_TIME:    "#2ecc71",
    UnwindVerdict.TOO_EARLY:  "#f1c40f",
    UnwindVerdict.TOO_LATE:   "#e74c3c",
}


def plot_steering_analysis(
    ref_states:   list,
    slow_states:  list,
    ref_arc:      np.ndarray,
    slow_arc:     np.ndarray,
    matches:      list[CornerMatch],
    ref_corners:  list[Corner],
    slow_corners: list[Corner],
    save_path:    str | None = None,
) -> None:
    """
    Three-panel steering plot:
      Top    : Raw steering traces + turn-in / apex / unwind markers
      Middle : |steering| traces + corner phase bands
      Bottom : Per-corner offset summary (bar chart)
    """
    ref_steer  = np.array([s.steering for s in ref_states])
    slow_steer = np.array([s.steering for s in slow_states])

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle("Steering Analysis — Reference (blue) vs Slow Lap (orange)",
                 fontsize=13, fontweight="bold")

    ax_raw, ax_abs, ax_bar = axes

    # ── Panel 1: Raw signed steering traces ──────────────────────────────────
    ax_raw.plot(ref_arc,  ref_steer,  color="#2980b9", lw=1.3, label="Reference", zorder=3)
    ax_raw.plot(slow_arc, slow_steer, color="#e67e22", lw=1.3, alpha=0.85,
                label="Slow lap", zorder=3)
    ax_raw.axhline(0, color="#aaa", lw=0.7, zorder=1)
    ax_raw.set_title("Steering angle (signed, rad)  ← left  /  right →",
                     fontsize=10, loc="left")
    ax_raw.set_ylabel("Steering (rad)")
    ax_raw.set_xlabel("Arc-length (m)")
    ax_raw.grid(True, alpha=0.3)

    # Mark turn-in / apex / unwind on the raw trace for each matched corner
    for m in matches:
        if m.slow is None:
            continue
        ti_col = _TI_COLORS[m.turn_in_verdict]
        ap_col = _AP_COLORS[m.apex_verdict]
        uw_col = _UW_COLORS[m.unwind_verdict]

        # Reference markers (solid)
        ax_raw.axvline(m.ref.turn_in_arc, color=ti_col, lw=1.2, ls="--", alpha=0.7, zorder=4)
        ax_raw.axvline(m.ref.apex_arc,    color=ap_col, lw=1.5, ls="-",  alpha=0.8, zorder=4)
        ax_raw.axvline(m.ref.unwind_arc,  color=uw_col, lw=1.2, ls=":",  alpha=0.7, zorder=4)

        # Slow lap markers (dotted, slightly offset for visibility)
        ax_raw.axvline(m.slow.turn_in_arc, color=ti_col, lw=1.0, ls="--", alpha=0.4, zorder=4)
        ax_raw.axvline(m.slow.apex_arc,    color=ap_col, lw=1.0, ls="-",  alpha=0.4, zorder=4)
        ax_raw.axvline(m.slow.unwind_arc,  color=uw_col, lw=1.0, ls=":",  alpha=0.4, zorder=4)

    legend_els = [
        mpatches.Patch(color="#2980b9", label="Reference"),
        mpatches.Patch(color="#e67e22", label="Slow lap"),
        plt.Line2D([0], [0], color="#555", ls="--", lw=1.2, label="Turn-in"),
        plt.Line2D([0], [0], color="#555", ls="-",  lw=1.5, label="Apex"),
        plt.Line2D([0], [0], color="#555", ls=":",  lw=1.2, label="Unwind"),
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="On time / similar"),
        mpatches.Patch(color="#e74c3c", alpha=0.5, label="Late / over-rotate"),
        mpatches.Patch(color="#f1c40f", alpha=0.5, label="Early / under-rotate"),
    ]
    ax_raw.legend(handles=legend_els, fontsize=7.5, loc="upper right", ncol=4)

    # ── Panel 2: |steering| + corner phase shading ───────────────────────────
    ax_abs.plot(ref_arc,  np.abs(ref_steer),  color="#2980b9", lw=1.3, label="Reference |steer|")
    ax_abs.plot(slow_arc, np.abs(slow_steer), color="#e67e22", lw=1.3, alpha=0.85,
                label="Slow |steer|")
    ax_abs.set_title("|Steering| with corner windows shaded", fontsize=10, loc="left")
    ax_abs.set_ylabel("|Steering| (rad)")
    ax_abs.set_xlabel("Arc-length (m)")
    ax_abs.grid(True, alpha=0.3)

    for m in matches:
        # Shade the reference corner window
        ax_abs.axvspan(m.ref.turn_in_arc, m.ref.unwind_arc,
                       alpha=0.12, color="#2980b9", zorder=1)
        if m.slow is not None:
            ax_abs.axvspan(m.slow.turn_in_arc, m.slow.unwind_arc,
                           alpha=0.12, color="#e67e22", zorder=1)

        # Apex dots
        ref_apex_val = abs(float(ref_steer[m.ref.apex_idx]))
        ax_abs.plot(m.ref.apex_arc, ref_apex_val, "o", color="#2980b9", ms=7, zorder=5)
        if m.slow is not None:
            slow_apex_val = abs(float(slow_steer[m.slow.apex_idx]))
            ax_abs.plot(m.slow.apex_arc, slow_apex_val, "o", color="#e67e22", ms=7, zorder=5)

    ax_abs.legend(fontsize=8, loc="upper right")

    # ── Panel 3: Offset bar chart ─────────────────────────────────────────────
    ax_bar.set_title(
        "Per-corner phase offsets  (slow − reference, metres)  "
        "positive = slow lap acts later on track",
        fontsize=10, loc="left",
    )

    valid = [m for m in matches if m.slow is not None]
    x     = np.arange(len(valid))
    width = 0.28

    ti_offsets  = [m.turn_in_offset_m  for m in valid]
    ap_offsets  = [m.apex_offset_m     for m in valid]
    uw_offsets  = [m.unwind_offset_m   for m in valid]

    bars_ti = ax_bar.bar(x - width, ti_offsets,  width, label="Turn-in Δ",  color="#3498db", alpha=0.85)
    bars_ap = ax_bar.bar(x,         ap_offsets,  width, label="Apex Δ",     color="#9b59b6", alpha=0.85)
    bars_uw = ax_bar.bar(x + width, uw_offsets,  width, label="Unwind Δ",   color="#1abc9c", alpha=0.85)

    ax_bar.axhline(0, color="#555", lw=0.8)
    ax_bar.set_ylabel("Offset (m)")
    ax_bar.set_xlabel("Corner (by reference apex position)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"{'L' if m.ref.direction > 0 else 'R'}\n{m.ref.apex_arc:.0f} m" for m in valid],
        fontsize=8,
    )
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.3)

    # Colour bars by verdict
    for bar, m in zip(bars_ti, valid):
        bar.set_facecolor(_TI_COLORS[m.turn_in_verdict])
    for bar, m in zip(bars_ap, valid):
        bar.set_facecolor(_AP_COLORS[m.apex_verdict])
    for bar, m in zip(bars_uw, valid):
        bar.set_facecolor(_UW_COLORS[m.unwind_verdict])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Steering analysis plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from parser import LapDataParser
    from parser import _arc_length

    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    ref_arc    = _arc_length(ref_states)
    ref_steer  = np.array([s.steering for s in ref_states])

    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()
    slow_arc    = _arc_length(slow_states)
    slow_steer  = np.array([s.steering for s in slow_states])

    print("Detecting corners …")
    ref_corners  = detect_corners(ref_states,  ref_arc)
    slow_corners = detect_corners(slow_states, slow_arc)
    print(f"  Reference corners : {len(ref_corners)}")
    print(f"  Slow lap corners  : {len(slow_corners)}")

    print("Matching corners …")
    matches = match_corners(
        ref_corners, slow_corners,
        ref_arc, slow_arc,
        ref_steer, slow_steer,
    )

    print_steering_recommendations(matches)

    save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_steering_analysis(
        ref_states, slow_states,
        ref_arc, slow_arc,
        matches, ref_corners, slow_corners,
        save_path=save,
    )