"""
gas_analysis.py
===============
Threshold-based throttle plateau analysis for lap comparison.

New logic
---------
We use a single global threshold for "gas is applied".

A throttle plateau is defined as a contiguous region where:
    gas >= applied_threshold

Events are derived from plateau boundaries:
  - plateau start: gas crosses threshold upward
  - plateau end:   gas crosses threshold downward

Comparison categories
---------------------
1. Timing misalignment
   Both laps contain the same plateau region in essence, but one starts and/or
   ends earlier/later than the reference.

2. Structural mismatch
   One lap contains a plateau where the other does not:
   - extra plateau in slow lap
   - missing plateau in slow lap
   - slow lap stops applying gas while reference still pushes
   - slow lap applies gas while reference does not

3. Level mismatch
   Both laps contain the plateau, but the gas level inside it differs
   materially (mean / peak level too low or too high).

Public API
----------
    detect_throttle_plateaus(states, arc, **kwargs) -> list[ThrottlePlateau]
    match_throttle_plateaus(ref, slow, **kwargs)    -> list[ThrottlePlateauMatch]
    find_extra_slow_plateaus(ref, slow, **kwargs)   -> list[ThrottlePlateau]
    print_gas_recommendations(matches, extra_slow)
    plot_gas_analysis(...)

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Uniform smoothing to reduce sensor noise before thresholding."""
    return uniform_filter1d(signal.astype(float), size=max(1, window))


def _arc_to_sample_distance(arc: np.ndarray, metres: float) -> int:
    """Convert a spatial distance in metres to an approximate sample count."""
    if len(arc) < 2 or arc[-1] <= arc[0]:
        return 1
    avg_spacing = (arc[-1] - arc[0]) / (len(arc) - 1)
    return max(1, int(metres / avg_spacing))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ThrottlePlateau:
    """
    A contiguous region where gas is considered applied.

    Start is the first sample where gas crosses the threshold upward.
    End is the last consecutive sample before it falls below threshold.
    """
    start_idx: int
    end_idx: int
    start_arc: float
    end_arc: float
    duration_m: float
    mean_gas: float
    peak_gas: float
    x_start: float
    y_start: float
    x_end: float
    y_end: float


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_throttle_plateaus(
    states: list,
    *,
    applied_threshold: float = 0.10,
    min_duration_m: float = 0.0,
    merge_gap_m: float = 10.0,
    smooth_window: int = 7,
) -> list[ThrottlePlateau]:
    """
    Detect threshold-based throttle plateaus.

    Plateau = contiguous region where gas >= applied_threshold.

    This version correctly handles edge plateaus:
    - if gas starts above threshold, a plateau begins at index 0
    - if gas ends above threshold, the plateau ends at the last index
    """
    if not states:
        return []

    gas = _smooth(np.array([s.gas for s in states]), smooth_window)
    # gas = np.array([s.gas for s in states])
    mask = gas >= applied_threshold
    n = len(mask)

    # Find contiguous True runs, including ones that start at 0 or end at n-1
    runs: list[tuple[int, int]] = []
    start: int | None = 0 if mask[0] else None

    for i in range(1, n):
        # False -> True : plateau starts at i
        if not mask[i - 1] and mask[i]:
            start = i
        # True -> False : plateau ends at i-1
        elif mask[i - 1] and not mask[i]:
            if start is not None:
                runs.append((start, i - 1))
                start = None

    # If still inside a plateau at the end, close it at the last sample
    if start is not None:
        runs.append((start, n - 1))

    if not runs:
        return []

    # Merge runs separated by very short gaps
    merged: list[tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap_m = float(states[s].arc - states[prev_e].arc)
        if gap_m <= merge_gap_m:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    plateaus: list[ThrottlePlateau] = []
    for s, e in merged:
        duration = float(states[e].arc - states[s].arc)
        if duration < min_duration_m:
            continue

        seg = gas[s:e + 1]
        plateaus.append(ThrottlePlateau(
            start_idx=s,
            end_idx=e,
            start_arc=float(states[s].arc),
            end_arc=float(states[e].arc),
            duration_m=duration,
            mean_gas=float(seg.mean()),
            peak_gas=float(seg.max()),
            x_start=states[s].x,
            y_start=states[s].y,
            x_end=states[e].x,
            y_end=states[e].y,
        ))

    return plateaus


# ---------------------------------------------------------------------------
# Matching / comparison of plateaus
# ---------------------------------------------------------------------------

class PlateauVerdict(Enum):
    MATCHED = auto()

    START_TOO_LATE = auto()
    START_TOO_EARLY = auto()
    END_TOO_LATE = auto()
    END_TOO_EARLY = auto()

    LEVEL_TOO_LOW = auto()
    LEVEL_TOO_HIGH = auto()

    MISSING = auto()
    EXTRA = auto()
    SPLIT = auto()
    MERGED = auto()


@dataclass
class PlateauBoundaryEvent:
    arc: float
    source: str            # "ref" | "slow"
    kind: str              # "start" | "end"
    plateau: ThrottlePlateau


@dataclass
class ThrottleBoundaryIssue:
    verdict: PlateauVerdict
    ref: ThrottlePlateau | None
    slow: ThrottlePlateau | None
    arc_start: float
    arc_end: float
    offset_m: float | None
    recommendation: str



def _gas_build_boundary_events(
    ref_plateaus: list[ThrottlePlateau],
    slow_plateaus: list[ThrottlePlateau],
) -> list[PlateauBoundaryEvent]:
    events: list[PlateauBoundaryEvent] = []

    for p in ref_plateaus:
        events.append(PlateauBoundaryEvent(
            arc=p.start_arc, source="ref", kind="start", plateau=p
        ))
        events.append(PlateauBoundaryEvent(
            arc=p.end_arc, source="ref", kind="end", plateau=p
        ))

    for p in slow_plateaus:
        events.append(PlateauBoundaryEvent(
            arc=p.start_arc, source="slow", kind="start", plateau=p
        ))
        events.append(PlateauBoundaryEvent(
            arc=p.end_arc, source="slow", kind="end", plateau=p
        ))

    # End before start at the same location, to make short lift/reapply behavior cleaner.
    kind_order = {"end": 0, "start": 1}
    src_order = {"ref": 0, "slow": 1}

    events.sort(key=lambda e: (e.arc, kind_order[e.kind], src_order[e.source]))
    return events



def analyze_throttle_boundaries(
    ref_plateaus: list[ThrottlePlateau],
    slow_plateaus: list[ThrottlePlateau],
    events: list[PlateauBoundaryEvent],
    *,
    timing_window_m: float = 12.0,
    timing_tolerance_m: float = 3.0,
    min_struct_gap_m: float = 2.0,
    level_mean_tol: float = 0.10,
    level_peak_tol: float = 0.12,
) -> list[ThrottleBoundaryIssue]:
    """
    Analyze throttle plateaus by sweeping through a single sorted list of
    plateau boundary events from both laps.

    Pairwise logic on adjacent events:
      ref start -> slow start   => MATCHED / START_TOO_LATE
      slow start -> ref start   => MATCHED / START_TOO_EARLY
      ref end   -> slow end     => MATCHED / END_TOO_LATE
      slow end  -> ref end      => MATCHED / END_TOO_EARLY

      slow start -> slow end, while ref inactive  => EXTRA
      ref  start -> ref  end, while slow inactive => MISSING

      slow end -> slow start, while ref active => SPLIT
      ref  end -> ref  start, while slow active => MERGED

    A same-kind cross-source boundary pair is considered MATCHED if the boundary
    distance is <= timing_tolerance_m.
    """
    if len(events) < 2:
        return []

    issues: list[ThrottleBoundaryIssue] = []

    # Active state in the open interval (e1.arc, e2.arc), after applying e1
    ref_active = False
    slow_active = False

    for i in range(len(events) - 1):
        e1 = events[i]
        e2 = events[i + 1]

        # Apply e1 so active flags describe the interval after e1
        if e1.source == "ref":
            ref_active = (e1.kind == "start")
        else:
            slow_active = (e1.kind == "start")

        gap = float(e2.arc - e1.arc)
        struct_gap_ok = gap >= min_struct_gap_m

        # ------------------------------------------------------------
        # Timing / matched boundaries:
        # same-kind, different-source, sufficiently close in arc
        # ------------------------------------------------------------
        if e1.kind == e2.kind and e1.source != e2.source and gap <= timing_window_m:
            is_matched = gap <= timing_tolerance_m

            if e1.kind == "start":
                if e1.source == "ref" and e2.source == "slow":
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MATCHED if is_matched else PlateauVerdict.START_TOO_LATE,
                        ref=e1.plateau,
                        slow=e2.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=gap,
                        recommendation=(
                            f"At ~{e1.arc:.0f} m your throttle start matches the reference "
                            f"well (offset {gap:.1f} m)."
                            if is_matched else
                            f"At ~{e1.arc:.0f} m the reference starts throttle, but you start "
                            f"at ~{e2.arc:.0f} m ({gap:.1f} m too late)."
                        ),
                    ))
                elif e1.source == "slow" and e2.source == "ref":
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MATCHED if is_matched else PlateauVerdict.START_TOO_EARLY,
                        ref=e2.plateau,
                        slow=e1.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=-gap,
                        recommendation=(
                            f"At ~{e2.arc:.0f} m your throttle start matches the reference "
                            f"well (offset {-gap:.1f} m)."
                            if is_matched else
                            f"You start throttle at ~{e1.arc:.0f} m, before the reference "
                            f"start at ~{e2.arc:.0f} m ({gap:.1f} m too early)."
                        ),
                    ))

            else:  # end/end
                if e1.source == "ref" and e2.source == "slow":
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MATCHED if is_matched else PlateauVerdict.END_TOO_LATE,
                        ref=e1.plateau,
                        slow=e2.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=gap,
                        recommendation=(
                            f"At ~{e1.arc:.0f} m your throttle end matches the reference "
                            f"well (offset {gap:.1f} m)."
                            if is_matched else
                            f"At ~{e1.arc:.0f} m the reference ends throttle, but you stay on "
                            f"until ~{e2.arc:.0f} m ({gap:.1f} m too late)."
                        ),
                    ))
                elif e1.source == "slow" and e2.source == "ref":
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MATCHED if is_matched else PlateauVerdict.END_TOO_EARLY,
                        ref=e2.plateau,
                        slow=e1.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=-gap,
                        recommendation=(
                            f"At ~{e2.arc:.0f} m your throttle end matches the reference "
                            f"well (offset {-gap:.1f} m)."
                            if is_matched else
                            f"You end throttle at ~{e1.arc:.0f} m, before the reference "
                            f"end at ~{e2.arc:.0f} m ({gap:.1f} m too early)."
                        ),
                    ))

        # ------------------------------------------------------------
        # Structural mismatches: adjacent same-source events
        # ------------------------------------------------------------
        if e1.source == e2.source and struct_gap_ok:
            # source start -> source end
            if e1.kind == "start" and e2.kind == "end":
                if e1.source == "slow" and not ref_active:
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.EXTRA,
                        ref=None,
                        slow=e1.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=None,
                        recommendation=(
                            f"Between {e1.arc:.0f}–{e2.arc:.0f} m you apply throttle, "
                            f"but the reference does not."
                        ),
                    ))
                elif e1.source == "ref" and not slow_active:
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MISSING,
                        ref=e1.plateau,
                        slow=None,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=None,
                        recommendation=(
                            f"Between {e1.arc:.0f}–{e2.arc:.0f} m the reference applies "
                            f"throttle, but you do not."
                        ),
                    ))

            # source end -> source start
            elif e1.kind == "end" and e2.kind == "start":
                if e1.source == "slow" and ref_active:
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.SPLIT,
                        ref=None,
                        slow=e2.plateau,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=None,
                        recommendation=(
                            f"Between {e1.arc:.0f}–{e2.arc:.0f} m you lift and reapply "
                            f"throttle, while the reference stays on throttle."
                        ),
                    ))
                elif e1.source == "ref" and slow_active:
                    issues.append(ThrottleBoundaryIssue(
                        verdict=PlateauVerdict.MERGED,
                        ref=e2.plateau,
                        slow=None,
                        arc_start=e1.arc,
                        arc_end=e2.arc,
                        offset_m=None,
                        recommendation=(
                            f"Between {e1.arc:.0f}–{e2.arc:.0f} the reference lifts and "
                            f"reapplies throttle, but you stay on throttle continuously."
                        ),
                    ))

    return issues

_ISSUE_ICONS = {
    PlateauVerdict.MATCHED: "✅",
    PlateauVerdict.START_TOO_LATE: "⚠️ ",
    PlateauVerdict.START_TOO_EARLY: "⚠️ ",
    PlateauVerdict.END_TOO_LATE: "⚠️ ",
    PlateauVerdict.END_TOO_EARLY: "⚠️ ",
    PlateauVerdict.LEVEL_TOO_LOW: "⚠️ ",
    PlateauVerdict.LEVEL_TOO_HIGH: "⚠️ ",
    PlateauVerdict.MISSING: "❌",
    PlateauVerdict.EXTRA: "⚠️ ",
    PlateauVerdict.SPLIT: "⚠️ ",
    PlateauVerdict.MERGED: "⚠️ ",
}



# ---------------------------------------------------------------------------
# Gas level analysis inside plateaus
# ---------------------------------------------------------------------------

@dataclass
class ThrottleLevelIssue:
    verdict: PlateauVerdict   # LEVEL_TOO_LOW or LEVEL_TOO_HIGH
    arc_start: float
    arc_end: float
    mean_delta: float         # slow_gas - ref_gas over the region
    recommendation: str


def _merge_level_regions(
    regions: list[tuple[float, float, str, float]],
    *,
    merge_gap_m: float,
) -> list[tuple[float, float, str, float]]:
    """
    Merge adjacent same-label regions separated by a small gap.

    Each region is:
        (start_arc, end_arc, label, mean_delta)
    where label is "low" or "high".
    """
    if not regions:
        return []

    merged = [regions[0]]

    for s, e, label, mean_delta in regions[1:]:
        prev_s, prev_e, prev_label, prev_mean = merged[-1]
        gap = s - prev_e

        if label == prev_label and gap <= merge_gap_m:
            # Weighted average by region length
            prev_len = max(prev_e - prev_s, 1e-9)
            curr_len = max(e - s, 1e-9)
            new_mean = (prev_mean * prev_len + mean_delta * curr_len) / (prev_len + curr_len)
            merged[-1] = (prev_s, e, label, new_mean)
        else:
            merged.append((s, e, label, mean_delta))

    return merged


def analyze_throttle_levels_in_mutual_plateaus(
    ref_states: list,
    slow_states: list,
    events: list[PlateauBoundaryEvent],
    *,
    gas_tolerance: float = 0.08,
    min_region_m: float = 5.0,
    merge_gap_m: float = 3.0,
) -> list[ThrottleLevelIssue]:
    """
    Analyze throttle level inside regions where BOTH laps are applying throttle.

    Procedure
    ---------
    1. Build one sorted boundary-event list from both laps.
    2. Sweep adjacent event pairs.
    3. Whenever the open interval between two adjacent events has both ref and slow
       throttle active, that interval is a mutual-throttle region.
    4. Inside that region, compare slow gas against ref gas on a common arc grid
       (union of ref and slow sample arcs, with interpolation).
    5. Split the region into contiguous subregions where slow gas is:
         - too low  (slow - ref < -gas_tolerance)
         - too high (slow - ref > +gas_tolerance)
       and ignore matched parts.
    """
    if len(events) < 2:
        return []

    ref_arc = np.array([s.arc for s in ref_states], dtype=float)
    ref_gas = np.array([s.gas for s in ref_states], dtype=float)
    slow_arc = np.array([s.arc for s in slow_states], dtype=float)
    slow_gas = np.array([s.gas for s in slow_states], dtype=float)

    issues: list[ThrottleLevelIssue] = []

    ref_active = False
    slow_active = False

    for i in range(len(events) - 1):
        e1 = events[i]
        e2 = events[i + 1]

        # Apply e1 so the flags describe the interval (e1.arc, e2.arc)
        if e1.source == "ref":
            ref_active = (e1.kind == "start")
        else:
            slow_active = (e1.kind == "start")

        region_start = float(e1.arc)
        region_end = float(e2.arc)

        if region_end <= region_start:
            continue

        # Only analyze regions where both are on throttle
        if not (ref_active and slow_active):
            continue

        # Build common comparison grid inside this mutual-throttle interval
        ref_mask = (ref_arc >= region_start) & (ref_arc <= region_end)
        slow_mask = (slow_arc >= region_start) & (slow_arc <= region_end)

        grid = np.concatenate([
            np.array([region_start, region_end], dtype=float),
            ref_arc[ref_mask],
            slow_arc[slow_mask],
        ])
        grid = np.unique(np.sort(grid))

        # Need at least one segment
        if len(grid) < 2:
            continue

        # Compare on interval midpoints rather than just nodes
        mids = 0.5 * (grid[:-1] + grid[1:])
        ref_mid = np.interp(mids, ref_arc, ref_gas)
        slow_mid = np.interp(mids, slow_arc, slow_gas)
        delta_mid = slow_mid - ref_mid

        # Label each small segment
        segment_labels: list[str] = []
        for d in delta_mid:
            if d < -gas_tolerance:
                segment_labels.append("low")
            elif d > gas_tolerance:
                segment_labels.append("high")
            else:
                segment_labels.append("matched")

        # Convert contiguous low/high segments into regions
        raw_regions: list[tuple[float, float, str, float]] = []
        curr_label = None
        curr_start = None
        curr_deltas: list[float] = []

        for j, label in enumerate(segment_labels):
            seg_start = float(grid[j])
            seg_end = float(grid[j + 1])

            if label == "matched":
                if curr_label is not None:
                    raw_regions.append((
                        curr_start,
                        seg_start,
                        curr_label,
                        float(np.mean(curr_deltas)),
                    ))
                    curr_label = None
                    curr_start = None
                    curr_deltas = []
                continue

            if curr_label is None:
                curr_label = label
                curr_start = seg_start
                curr_deltas = [float(delta_mid[j])]
            elif curr_label == label:
                curr_deltas.append(float(delta_mid[j]))
            else:
                raw_regions.append((
                    curr_start,
                    seg_start,
                    curr_label,
                    float(np.mean(curr_deltas)),
                ))
                curr_label = label
                curr_start = seg_start
                curr_deltas = [float(delta_mid[j])]

        if curr_label is not None:
            raw_regions.append((
                curr_start,
                float(grid[-1]),
                curr_label,
                float(np.mean(curr_deltas)),
            ))

        # Merge nearby same-sign regions across tiny matched gaps
        merged_regions = _merge_level_regions(raw_regions, merge_gap_m=merge_gap_m)

        # Filter by size and create recommendations
        for s, e, label, mean_delta in merged_regions:
            duration = e - s
            if duration < min_region_m:
                continue

            if label == "low":
                issues.append(ThrottleLevelIssue(
                    verdict=PlateauVerdict.LEVEL_TOO_LOW,
                    arc_start=s,
                    arc_end=e,
                    mean_delta=mean_delta,
                    recommendation=(
                        f"Between {s:.0f}–{e:.0f} m you should apply more gas "
                        f"(your throttle is lower than the reference by about "
                        f"{abs(mean_delta):.0%} on average)."
                    ),
                ))
            elif label == "high":
                issues.append(ThrottleLevelIssue(
                    verdict=PlateauVerdict.LEVEL_TOO_HIGH,
                    arc_start=s,
                    arc_end=e,
                    mean_delta=mean_delta,
                    recommendation=(
                        f"Between {s:.0f}–{e:.0f} m you should apply less gas "
                        f"(your throttle is higher than the reference by about "
                        f"{abs(mean_delta):.0%} on average)."
                    ),
                ))

    return issues



# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_gas_recommendations(boundary_issues: list[ThrottleBoundaryIssue], level_issues: list[ThrottleLevelIssue]) -> None:
    print("\n" + "=" * 64)
    print("  GAS — Boundary Analysis")
    print("=" * 64)
    for issue in boundary_issues:
        icon = _ISSUE_ICONS.get(issue.verdict, "?")
        print(f"  {icon}  {issue.recommendation}")

    print("\n" + "=" * 64)
    print("  GAS — Throttle Level Inside Mutual Plateaus")
    print("=" * 64)
    for issue in level_issues:
        icon = "⚠️ " if issue.verdict in {PlateauVerdict.LEVEL_TOO_LOW, PlateauVerdict.LEVEL_TOO_HIGH} else "✅"
        print(f"  {icon}  {issue.recommendation}")



def plot_gas_analysis(
    ref_states: list,
    slow_states: list,
    ref_plateaus: list[ThrottlePlateau],
    slow_plateaus: list[ThrottlePlateau],
    boundary_issues: list[ThrottleBoundaryIssue],
    level_issues: list[ThrottleLevelIssue],
    save_path: str | None = None,
) -> None:
    """
    Two-panel gas plot using aligned CarState.arc.

    Top panel:
      - raw gas traces
      - reference / slow plateau spans
      - timing arrows for early/late/matched start/end boundaries

    Bottom panel:
      - structural mismatch regions (missing / extra / split / merged)
      - level mismatch regions inside mutual plateaus
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ref_arc = np.array([s.arc for s in ref_states], dtype=float)
    slow_arc = np.array([s.arc for s in slow_states], dtype=float)

    ref_gas = np.array([s.gas for s in ref_states], dtype=float)
    slow_gas = np.array([s.gas for s in slow_states], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    fig.suptitle("Gas / Throttle Analysis", fontsize=13, fontweight="bold")

    # ------------------------------------------------------------------
    # Common trace plotting
    # ------------------------------------------------------------------
    for ax in (ax1, ax2):
        ax.plot(ref_arc, ref_gas, color="#2980b9", lw=1.5, label="Reference (fast)", zorder=3)
        ax.plot(slow_arc, slow_gas, color="#e67e22", lw=1.4, alpha=0.9, label="Slow lap", zorder=3)
        ax.set_ylabel("Gas (0–1)", fontsize=9)
        ax.set_xlabel("Arc-length (m)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.08, 1.20)

    # ------------------------------------------------------------------
    # Top panel: plateaus + timing offsets
    # ------------------------------------------------------------------
    ax1.set_title("Throttle Plateaus and Boundary Timing", fontsize=10, loc="left")

    # Background plateau spans
    for p in ref_plateaus:
        ax1.axvspan(p.start_arc, p.end_arc, alpha=0.14, color="#2980b9", zorder=1)
        ax1.plot(p.start_arc, 1.04, "v", color="#2980b9", ms=7, zorder=5)
        ax1.plot(p.end_arc, 1.04, "s", color="#2980b9", ms=5, zorder=5)

    for p in slow_plateaus:
        ax1.axvspan(p.start_arc, p.end_arc, alpha=0.14, color="#e67e22", zorder=1)
        ax1.plot(p.start_arc, 0.98, "^", color="#e67e22", ms=7, zorder=5)
        ax1.plot(p.end_arc, 0.98, "o", color="#e67e22", ms=5, zorder=5)

    # Timing issue styling
    timing_colors = {
        PlateauVerdict.MATCHED: "#2ecc71",
        PlateauVerdict.START_TOO_LATE: "#e74c3c",
        PlateauVerdict.START_TOO_EARLY: "#f1c40f",
        PlateauVerdict.END_TOO_LATE: "#c0392b",
        PlateauVerdict.END_TOO_EARLY: "#f39c12",
    }

    timing_verdicts = {
        PlateauVerdict.MATCHED,
        PlateauVerdict.START_TOO_LATE,
        PlateauVerdict.START_TOO_EARLY,
        PlateauVerdict.END_TOO_LATE,
        PlateauVerdict.END_TOO_EARLY,
    }

    for issue in boundary_issues:
        if issue.verdict not in timing_verdicts:
            continue

        color = timing_colors[issue.verdict]

        # Start timing issues
        if issue.verdict in {
            PlateauVerdict.MATCHED,
            PlateauVerdict.START_TOO_LATE,
            PlateauVerdict.START_TOO_EARLY,
        }:
            if issue.ref is None or issue.slow is None:
                continue

            y = 1.11
            ref_x = issue.ref.start_arc
            slow_x = issue.slow.start_arc

            ax1.plot(ref_x, y, "v", color="#2980b9", ms=8, zorder=7)
            ax1.plot(slow_x, y, "^", color="#e67e22", ms=8, zorder=7)

            ax1.annotate(
                "",
                xy=(slow_x, y),
                xytext=(ref_x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=6,
            )

            mid = 0.5 * (ref_x + slow_x)
            label = (
                f"{issue.offset_m:+.0f} m"
                if issue.offset_m is not None
                else "0 m"
            )
            ax1.text(mid, y + 0.025, label, ha="center", fontsize=7, color=color, zorder=8)

        # End timing issues
        if issue.verdict in {
            PlateauVerdict.MATCHED,
            PlateauVerdict.END_TOO_LATE,
            PlateauVerdict.END_TOO_EARLY,
        }:
            if issue.ref is None or issue.slow is None:
                continue

            y = 1.15
            ref_x = issue.ref.end_arc
            slow_x = issue.slow.end_arc

            ax1.plot(ref_x, y, "s", color="#2980b9", ms=6, zorder=7)
            ax1.plot(slow_x, y, "o", color="#e67e22", ms=6, zorder=7)

            ax1.annotate(
                "",
                xy=(slow_x, y),
                xytext=(ref_x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=6,
            )

            mid = 0.5 * (ref_x + slow_x)
            label = (
                f"{issue.offset_m:+.0f} m"
                if issue.offset_m is not None
                else "0 m"
            )
            ax1.text(mid, y + 0.02, label, ha="center", fontsize=7, color=color, zorder=8)

    legend_top = [
        mpatches.Patch(color="#2980b9", alpha=0.35, label="Reference plateau"),
        mpatches.Patch(color="#e67e22", alpha=0.35, label="Slow plateau"),
        plt.Line2D([0], [0], color="#2ecc71", lw=2, label="Boundary matched"),
        plt.Line2D([0], [0], color="#e74c3c", lw=2, label="Start too late"),
        plt.Line2D([0], [0], color="#f1c40f", lw=2, label="Start too early"),
        plt.Line2D([0], [0], color="#c0392b", lw=2, label="End too late"),
        plt.Line2D([0], [0], color="#f39c12", lw=2, label="End too early"),
    ]
    ax1.legend(handles=legend_top, fontsize=7.5, loc="upper right", ncol=3)

    # ------------------------------------------------------------------
    # Bottom panel: structural + level issues
    # ------------------------------------------------------------------
    ax2.set_title("Structure and Throttle-Level Mismatches", fontsize=10, loc="left")

    # Show plateau spans faintly again for context
    for p in ref_plateaus:
        ax2.axvspan(p.start_arc, p.end_arc, alpha=0.06, color="#2980b9", zorder=1)
    for p in slow_plateaus:
        ax2.axvspan(p.start_arc, p.end_arc, alpha=0.06, color="#e67e22", zorder=1)

    structure_colors = {
        PlateauVerdict.MISSING: "#7f8c8d",
        PlateauVerdict.EXTRA: "#34495e",
        PlateauVerdict.SPLIT: "#8e44ad",
        PlateauVerdict.MERGED: "#1abc9c",
    }

    # Plot structural issues as bold shaded regions
    for issue in boundary_issues:
        if issue.verdict not in structure_colors:
            continue

        color = structure_colors[issue.verdict]
        ax2.axvspan(issue.arc_start, issue.arc_end, alpha=0.28, color=color, zorder=2)

        mid = 0.5 * (issue.arc_start + issue.arc_end)
        label_map = {
            PlateauVerdict.MISSING: "missing",
            PlateauVerdict.EXTRA: "extra",
            PlateauVerdict.SPLIT: "split",
            PlateauVerdict.MERGED: "merged",
        }
        ax2.text(
            mid,
            1.08,
            label_map[issue.verdict],
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            zorder=5,
        )

    # Plot level issues as colored subregions inside mutual plateaus
    level_colors = {
        PlateauVerdict.LEVEL_TOO_LOW: "#9b59b6",
        PlateauVerdict.LEVEL_TOO_HIGH: "#16a085",
    }

    for issue in level_issues:
        if issue.verdict not in level_colors:
            continue

        color = level_colors[issue.verdict]
        ax2.axvspan(issue.arc_start, issue.arc_end, ymin=0.05, ymax=0.92, alpha=0.22, color=color, zorder=3)

        mid = 0.5 * (issue.arc_start + issue.arc_end)
        short = "more gas" if issue.verdict == PlateauVerdict.LEVEL_TOO_LOW else "less gas"
        ax2.text(
            mid,
            0.10,
            short,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            rotation=0,
            zorder=6,
        )

    legend_bottom = [
        mpatches.Patch(color=structure_colors[PlateauVerdict.MISSING], alpha=0.5, label="Missing plateau"),
        mpatches.Patch(color=structure_colors[PlateauVerdict.EXTRA], alpha=0.5, label="Extra plateau"),
        mpatches.Patch(color=structure_colors[PlateauVerdict.SPLIT], alpha=0.5, label="Split plateau"),
        mpatches.Patch(color=structure_colors[PlateauVerdict.MERGED], alpha=0.5, label="Merged plateau"),
        mpatches.Patch(color=level_colors[PlateauVerdict.LEVEL_TOO_LOW], alpha=0.4, label="Apply more gas"),
        mpatches.Patch(color=level_colors[PlateauVerdict.LEVEL_TOO_HIGH], alpha=0.4, label="Apply less gas"),
    ]
    ax2.legend(handles=legend_bottom, fontsize=7.5, loc="upper right", ncol=3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gas analysis plot saved → {save_path}")
    else:
        plt.show()




if __name__ == "__main__":
    import sys
    from parser import LapDataParser, align_laps, filter_arc_jumps

    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()

    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()

    ref_states, slow_states = align_laps(ref_states, slow_states)

    ref_count = len(ref_states)
    slow_count = len(slow_states)

    ref_states = filter_arc_jumps(ref_states)
    slow_states = filter_arc_jumps(slow_states)

    print(f"{ref_count - len(ref_states)} ref points discarded")
    print(f"{slow_count - len(slow_states)} slot points discarded")

    ref_plateaus = detect_throttle_plateaus(ref_states)
    slow_plateaus = detect_throttle_plateaus(slow_states)

    print("Ref plateaus:")
    for pl in ref_plateaus:
        print(f"{pl.start_arc} - {pl.end_arc}")

    print("Slow plateaus:")
    for pl in slow_plateaus:
        print(f"{pl.start_arc} - {pl.end_arc}")

    events = _gas_build_boundary_events(ref_plateaus, slow_plateaus)

    boundary_issues = analyze_throttle_boundaries(
        ref_plateaus,
        slow_plateaus,
        events,
    )

    level_issues = analyze_throttle_levels_in_mutual_plateaus(
        ref_states,
        slow_states,
        events,
    )

    print_gas_recommendations(boundary_issues, level_issues)

    save = sys.argv[1] if len(sys.argv) > 1 else None

    plot_gas_analysis(
        ref_states,
        slow_states,
        ref_plateaus,
        slow_plateaus,
        boundary_issues,
        level_issues,
    )