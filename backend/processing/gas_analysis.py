"""
===============
Throttle-specific event detection for lap comparison.

Throttle (gas) is a *sustained* signal, not a spike.  The meaningful events are:

  - **ThrottleApplication**: the moment the driver commits to throttle after a
    corner — detected as a rising edge where gas crosses a low threshold going
    upward.  Directly comparable across laps: "you opened the throttle 18 m late."

  - **ThrottlePlateau**: a sustained high-throttle region (driver fully committed).
    Comparable by start position, duration, and mean level:
    "you lifted 30 m before the reference" / "you never fully committed here."

Both are expressed in arc-length (metres) so they are speed-independent.

Public API
----------
    detect_throttle_applications(states, arc, **kwargs) -> list[ThrottleApplication]
    detect_throttle_plateaus(states, arc, **kwargs)     -> list[ThrottlePlateau]
    match_throttle_applications(ref, slow, ...)         -> list[ThrottleAppMatch]
    match_throttle_plateaus(ref, slow, ...)             -> list[ThrottlePlateauMatch]
    print_gas_recommendations(app_matches, plat_matches)

Dependencies: numpy, scipy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Uniform (box) smoothing to remove sensor noise before differentiation."""
    return uniform_filter1d(signal.astype(float), size=max(1, window))


def _arc_to_sample_distance(arc: np.ndarray, metres: float) -> int:
    """Convert a spatial distance in metres to an approximate sample count."""
    if len(arc) < 2 or arc[-1] <= arc[0]:
        return 1
    avg_spacing = (arc[-1] - arc[0]) / (len(arc) - 1)
    return max(1, int(metres / avg_spacing))


# ---------------------------------------------------------------------------
# ThrottleApplication — rising-edge events
# ---------------------------------------------------------------------------

@dataclass
class ThrottleApplication:
    """
    The moment a driver commits to throttle after a corner.

    Detected as a rising edge: gas crosses `low_threshold` going upward,
    and subsequently reaches `high_threshold` within `confirm_window_m` metres
    (confirming it's a genuine application, not a flutter).
    """
    index: int          # sample index of the rising edge
    arc_pos: float      # arc-length at rising edge (m)
    peak_value: float   # maximum gas reached in the following confirm window
    peak_arc: float     # arc-length of that peak
    x: float
    y: float


def detect_throttle_applications(
    states: list,           # list[CarState]
    arc: np.ndarray,
    *,
    low_threshold: float  = 0.10,   # gas level that defines "off throttle"
    high_threshold: float = 0.55,   # gas level that must be reached to confirm
    confirm_window_m: float = 40.0, # metres within which high_threshold must be hit
    min_gap_m: float = 30.0,        # minimum metres between two events
    smooth_window: int = 7,         # samples for pre-smoothing
) -> list[ThrottleApplication]:
    """
    Detect throttle-application events (rising edges) in *states*.

    Algorithm
    ---------
    1. Smooth the gas signal to remove noise.
    2. Compute the discrete derivative (Δgas / sample).
    3. Mark candidate edges: samples where gas crosses low_threshold upward
       AND the derivative is positive.
    4. For each candidate, scan forward up to confirm_window_m metres to check
       that gas actually reaches high_threshold.  If it does, record an event.
    5. Enforce minimum gap: suppress any new event within min_gap_m of the last.
    """
    gas = _smooth(np.array([s.gas for s in states]), smooth_window)
    n = len(gas)
    events: list[ThrottleApplication] = []
    last_event_arc = -min_gap_m - 1.0

    confirm_samples = _arc_to_sample_distance(arc, confirm_window_m)
    min_gap_samples = _arc_to_sample_distance(arc, min_gap_m)

    for i in range(1, n):
        # Rising edge: was below threshold, now at or above it
        if gas[i - 1] < low_threshold <= gas[i]:
            # Enforce minimum gap
            if arc[i] - last_event_arc < min_gap_m:
                continue

            # Look ahead to confirm the application is genuine
            end = min(i + confirm_samples, n)
            window_gas = gas[i:end]

            if window_gas.max() >= high_threshold:
                peak_local = int(np.argmax(window_gas))
                peak_idx   = i + peak_local

                events.append(ThrottleApplication(
                    index=i,
                    arc_pos=float(arc[i]),
                    peak_value=float(window_gas[peak_local]),
                    peak_arc=float(arc[peak_idx]),
                    x=states[i].x,
                    y=states[i].y,
                ))
                last_event_arc = float(arc[i])

    return events


# ---------------------------------------------------------------------------
# ThrottlePlateau — sustained high-throttle regions
# ---------------------------------------------------------------------------

@dataclass
class ThrottlePlateau:
    """
    A sustained high-throttle region (driver fully committed).

    Characterised by start, end, duration (metres), and mean gas level.
    """
    start_idx: int
    end_idx: int
    start_arc: float    # m
    end_arc: float      # m
    duration_m: float   # end_arc - start_arc
    mean_gas: float
    x_start: float
    y_start: float


def detect_throttle_plateaus(
    states: list,
    arc: np.ndarray,
    *,
    plateau_threshold: float = 0.70,  # gas level to count as "full throttle"
    min_duration_m: float = 20.0,     # minimum length to count as a plateau
    merge_gap_m: float = 10.0,        # merge two plateaus if gap < this
    smooth_window: int = 9,
) -> list[ThrottlePlateau]:
    """
    Detect sustained high-throttle plateaus.

    Algorithm
    ---------
    1. Smooth and threshold the gas signal → binary "in plateau" mask.
    2. Find contiguous True runs in the mask.
    3. Merge runs separated by less than merge_gap_m (brief lifts in a long
       straight, e.g. gear changes).
    4. Keep only runs whose arc-length span ≥ min_duration_m.
    """
    gas = _smooth(np.array([s.gas for s in states]), smooth_window)
    mask = gas >= plateau_threshold

    # Find contiguous runs
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))

    if not runs:
        return []

    # Merge close runs
    merge_gap_samples = _arc_to_sample_distance(arc, merge_gap_m)
    merged: list[tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filter by minimum duration and build dataclasses
    plateaus: list[ThrottlePlateau] = []
    for s, e in merged:
        duration = arc[e] - arc[s]
        if duration >= min_duration_m:
            plateaus.append(ThrottlePlateau(
                start_idx=s,
                end_idx=e,
                start_arc=float(arc[s]),
                end_arc=float(arc[e]),
                duration_m=float(duration),
                mean_gas=float(gas[s:e + 1].mean()),
                x_start=states[s].x,
                y_start=states[s].y,
            ))

    return plateaus


# ---------------------------------------------------------------------------
# Matching — ThrottleApplication
# ---------------------------------------------------------------------------

class AppVerdict(Enum):
    ON_TIME   = auto()
    TOO_LATE  = auto()   # slow driver commits to throttle later on track
    TOO_EARLY = auto()   # slow driver commits earlier (unusual but possible)
    MISSING   = auto()   # no matching application in search window


@dataclass
class ThrottleAppMatch:
    ref: ThrottleApplication
    slow: ThrottleApplication | None
    verdict: AppVerdict
    offset_m: float           # slow.arc_pos − ref.arc_pos (+  = later)
    recommendation: str


def _nearest_app(
    target_arc: float,
    candidates: list[ThrottleApplication],
    radius_m: float,
) -> ThrottleApplication | None:
    within = [a for a in candidates if abs(a.arc_pos - target_arc) <= radius_m]
    return min(within, key=lambda a: abs(a.arc_pos - target_arc)) if within else None


def match_throttle_applications(
    ref_apps: list[ThrottleApplication],
    slow_apps: list[ThrottleApplication],
    *,
    on_time_window_m: float = 10.0,
    search_radius_m: float  = 60.0,
) -> list[ThrottleAppMatch]:
    """Match each reference throttle-application to the nearest slow-lap equivalent."""
    results: list[ThrottleAppMatch] = []

    for ref in ref_apps:
        slow = _nearest_app(ref.arc_pos, slow_apps, search_radius_m)

        if slow is None:
            verdict = AppVerdict.MISSING
            offset  = float("nan")
            rec = (
                f"At ~{ref.arc_pos:.0f} m the reference driver applies throttle "
                f"(peaks at {ref.peak_value:.0%}) — "
                f"no matching throttle application found within "
                f"{search_radius_m:.0f} m of this position."
            )
        else:
            offset = slow.arc_pos - ref.arc_pos
            if abs(offset) <= on_time_window_m:
                verdict = AppVerdict.ON_TIME
                rec = (
                    f"At ~{ref.arc_pos:.0f} m your throttle timing is good "
                    f"(offset {offset:+.1f} m).  "
                    f"Reference peak: {ref.peak_value:.0%}, yours: {slow.peak_value:.0%}."
                )
            elif offset > 0:
                # slow driver opens throttle later (further along track)
                verdict = AppVerdict.TOO_LATE
                rec = (
                    f"At ~{ref.arc_pos:.0f} m you should get on the throttle sooner — "
                    f"you opened it {abs(offset):.1f} m too late "
                    f"(you: {slow.arc_pos:.0f} m, reference: {ref.arc_pos:.0f} m).  "
                    f"Getting on gas earlier here will carry more exit speed."
                )
            else:
                verdict = AppVerdict.TOO_EARLY
                rec = (
                    f"At ~{ref.arc_pos:.0f} m you applied throttle "
                    f"{abs(offset):.1f} m earlier than the reference "
                    f"(you: {slow.arc_pos:.0f} m, reference: {ref.arc_pos:.0f} m).  "
                    f"This may be causing understeer on corner exit — "
                    f"wait until you're more settled before committing."
                )

        results.append(ThrottleAppMatch(
            ref=ref, slow=slow, verdict=verdict,
            offset_m=offset, recommendation=rec,
        ))

    return results


# ---------------------------------------------------------------------------
# Matching — ThrottlePlateau
# ---------------------------------------------------------------------------

class PlateauVerdict(Enum):
    MATCHED      = auto()   # similar start, similar duration
    LATE_START   = auto()   # slow driver starts plateau later
    EARLY_LIFT   = auto()   # slow driver ends plateau earlier (shorter)
    UNDER_COMMIT = auto()   # slow driver never reaches plateau_threshold here
    MISSING      = auto()   # no overlapping plateau found


@dataclass
class ThrottlePlateauMatch:
    ref: ThrottlePlateau
    slow: ThrottlePlateau | None
    verdict: PlateauVerdict
    start_offset_m: float     # slow.start_arc − ref.start_arc
    duration_delta_m: float   # slow.duration_m − ref.duration_m (negative = shorter)
    recommendation: str


def _overlapping_plateau(
    ref: ThrottlePlateau,
    candidates: list[ThrottlePlateau],
    search_radius_m: float,
) -> ThrottlePlateau | None:
    """Return the candidate plateau whose start is closest to ref.start_arc."""
    within = [
        p for p in candidates
        if abs(p.start_arc - ref.start_arc) <= search_radius_m
        or (p.start_arc <= ref.end_arc and p.end_arc >= ref.start_arc)
    ]
    return min(within, key=lambda p: abs(p.start_arc - ref.start_arc)) if within else None


def match_throttle_plateaus(
    ref_plateaus: list[ThrottlePlateau],
    slow_plateaus: list[ThrottlePlateau],
    *,
    on_time_window_m: float   = 10.0,
    search_radius_m: float    = 60.0,
    duration_threshold_m: float = 15.0,  # delta duration to flag "early lift"
) -> list[ThrottlePlateauMatch]:
    results: list[ThrottlePlateauMatch] = []

    for ref in ref_plateaus:
        slow = _overlapping_plateau(ref, slow_plateaus, search_radius_m)

        if slow is None:
            verdict = PlateauVerdict.MISSING
            s_off = float("nan")
            d_off = float("nan")
            rec = (
                f"Between {ref.start_arc:.0f}–{ref.end_arc:.0f} m the reference "
                f"holds full throttle ({ref.duration_m:.0f} m, "
                f"avg {ref.mean_gas:.0%}) — "
                f"no matching full-throttle region found. "
                f"You may be lifting where you should be committed."
            )
        else:
            s_off = slow.start_arc - ref.start_arc
            d_off = slow.duration_m - ref.duration_m

            issues: list[str] = []

            if s_off > on_time_window_m:
                verdict = PlateauVerdict.LATE_START
                issues.append(
                    f"you opened throttle {s_off:.0f} m late "
                    f"(get on gas at ~{ref.start_arc:.0f} m, not {slow.start_arc:.0f} m)"
                )
            elif s_off < -on_time_window_m:
                # Rare — driver opened throttle early but maybe didn't hold it
                issues.append(
                    f"you opened throttle {abs(s_off):.0f} m early — check for understeer"
                )

            if d_off < -duration_threshold_m:
                verdict = PlateauVerdict.EARLY_LIFT
                issues.append(
                    f"you lifted {abs(d_off):.0f} m before the reference "
                    f"(held {slow.duration_m:.0f} m vs {ref.duration_m:.0f} m) — "
                    f"trust the grip and stay on it longer"
                )

            if slow.mean_gas < ref.mean_gas - 0.10:
                verdict = PlateauVerdict.UNDER_COMMIT
                issues.append(
                    f"your average throttle ({slow.mean_gas:.0%}) is well below "
                    f"the reference ({ref.mean_gas:.0%}) — commit more fully"
                )

            if not issues:
                verdict = PlateauVerdict.MATCHED
                rec = (
                    f"Between {ref.start_arc:.0f}–{ref.end_arc:.0f} m "
                    f"your full-throttle application matches the reference well."
                )
            else:
                rec = (
                    f"Between {ref.start_arc:.0f}–{ref.end_arc:.0f} m: "
                    + "; ".join(issues) + "."
                )

        results.append(ThrottlePlateauMatch(
            ref=ref, slow=slow, verdict=verdict,
            start_offset_m=s_off, duration_delta_m=d_off,
            recommendation=rec,
        ))

    return results


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_APP_ICONS = {
    AppVerdict.ON_TIME:   "✅",
    AppVerdict.TOO_LATE:  "⚠️ ",
    AppVerdict.TOO_EARLY: "⚠️ ",
    AppVerdict.MISSING:   "❌",
}

_PLAT_ICONS = {
    PlateauVerdict.MATCHED:      "✅",
    PlateauVerdict.LATE_START:   "⚠️ ",
    PlateauVerdict.EARLY_LIFT:   "⚠️ ",
    PlateauVerdict.UNDER_COMMIT: "⚠️ ",
    PlateauVerdict.MISSING:      "❌",
}


def print_gas_recommendations(
    app_matches: list[ThrottleAppMatch],
    plat_matches: list[ThrottlePlateauMatch],
) -> None:
    print("\n" + "=" * 60)
    print("  GAS — Throttle Application Timing")
    print("=" * 60)
    for m in app_matches:
        icon = _APP_ICONS.get(m.verdict, "?")
        print(f"  {icon}  {m.recommendation}")

    print("\n" + "=" * 60)
    print("  GAS — Full-Throttle Plateau Comparison")
    print("=" * 60)
    for m in plat_matches:
        icon = _PLAT_ICONS.get(m.verdict, "?")
        print(f"  {icon}  {m.recommendation}")


# ---------------------------------------------------------------------------
# Plotting helpers (callable from lap_analysis.py)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_APP_COLORS = {
    AppVerdict.ON_TIME:   "#2ecc71",
    AppVerdict.TOO_LATE:  "#e74c3c",
    AppVerdict.TOO_EARLY: "#f1c40f",
    AppVerdict.MISSING:   "#95a5a6",
}

_PLAT_COLORS = {
    PlateauVerdict.MATCHED:      "#2ecc71",
    PlateauVerdict.LATE_START:   "#e74c3c",
    PlateauVerdict.EARLY_LIFT:   "#e67e22",
    PlateauVerdict.UNDER_COMMIT: "#f1c40f",
    PlateauVerdict.MISSING:      "#95a5a6",
}


def plot_gas_analysis(
    ref_states: list,
    slow_states: list,
    ref_arc: np.ndarray,
    slow_arc: np.ndarray,
    app_matches: list[ThrottleAppMatch],
    plat_matches: list[ThrottlePlateauMatch],
    ref_apps: list[ThrottleApplication],
    slow_apps: list[ThrottleApplication],
    ref_plateaus: list[ThrottlePlateau],
    slow_plateaus: list[ThrottlePlateau],
    save_path: str | None = None,
) -> None:
    """
    Two-panel gas plot:
      Top:    Raw traces + throttle-application rising-edge markers
      Bottom: Raw traces + full-throttle plateau bands
    """
    ref_gas  = np.array([s.gas for s in ref_states])
    slow_gas = np.array([s.gas for s in slow_states])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle("Gas / Throttle Analysis", fontsize=13, fontweight="bold")

    for ax in (ax1, ax2):
        ax.plot(ref_arc,  ref_gas,  color="#2980b9", lw=1.4, label="Reference (fast)", zorder=3)
        ax.plot(slow_arc, slow_gas, color="#e67e22", lw=1.4, alpha=0.85, label="Slow lap", zorder=3)
        ax.set_ylabel("Gas (0–1)", fontsize=9)
        ax.set_xlabel("Arc-length (m)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    # --- Top panel: rising-edge (application) events ---
    ax1.set_title("Throttle Application Timing  (▶ = commit point)", fontsize=10, loc="left")

    for m in app_matches:
        color = _APP_COLORS[m.verdict]
        # Vertical line at reference commit point
        ax1.axvline(m.ref.arc_pos, color=color, lw=1.5, alpha=0.8, zorder=4)
        ax1.plot(m.ref.arc_pos, m.ref.peak_value, "v", color=color, ms=9, zorder=5,
                 label=f"Ref commit ({m.verdict.name})")
        if m.slow is not None:
            ax1.plot(m.slow.arc_pos, m.slow.peak_value, "^", color="#e67e22",
                     ms=8, zorder=5)
            # Arrow showing the offset
            ax1.annotate(
                "",
                xy=(m.slow.arc_pos, 1.05),
                xytext=(m.ref.arc_pos, 1.05),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=6,
            )
            mid = (m.ref.arc_pos + m.slow.arc_pos) / 2
            ax1.text(mid, 1.08, f"{m.offset_m:+.0f} m", ha="center",
                     fontsize=7, color=color, zorder=7)

    legend_els = [
        mpatches.Patch(color="#2980b9", label="Reference"),
        mpatches.Patch(color="#e67e22", label="Slow lap"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#2ecc71", ms=9, label="On time"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c", ms=9, label="Too late"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#f1c40f", ms=9, label="Too early"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#95a5a6", ms=9, label="Missing"),
    ]
    ax1.legend(handles=legend_els, fontsize=7.5, loc="upper right", ncol=3)

    # --- Bottom panel: plateau bands ---
    ax2.set_title("Full-Throttle Plateaus  (shaded regions)", fontsize=10, loc="left")

    for p in ref_plateaus:
        ax2.axvspan(p.start_arc, p.end_arc, alpha=0.18, color="#2980b9", zorder=1)

    for p in slow_plateaus:
        ax2.axvspan(p.start_arc, p.end_arc, alpha=0.18, color="#e67e22", zorder=1)

    for m in plat_matches:
        color = _PLAT_COLORS[m.verdict]
        # Mark reference plateau start with a triangle
        ypos = 1.02
        ax2.plot(m.ref.start_arc, ypos, "v", color="#2980b9", ms=9, zorder=5)
        if m.slow is not None:
            ax2.plot(m.slow.start_arc, ypos, "^", color="#e67e22", ms=8, zorder=5)
            if abs(m.start_offset_m) > 5:
                ax2.annotate(
                    "",
                    xy=(m.slow.start_arc, ypos),
                    xytext=(m.ref.start_arc, ypos),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=6,
                )

    legend_els2 = [
        mpatches.Patch(color="#2980b9", alpha=0.4, label="Reference plateau"),
        mpatches.Patch(color="#e67e22", alpha=0.4, label="Slow plateau"),
        mpatches.Patch(color=_PLAT_COLORS[PlateauVerdict.MATCHED],      label="Matched"),
        mpatches.Patch(color=_PLAT_COLORS[PlateauVerdict.LATE_START],   label="Late start"),
        mpatches.Patch(color=_PLAT_COLORS[PlateauVerdict.EARLY_LIFT],   label="Early lift"),
        mpatches.Patch(color=_PLAT_COLORS[PlateauVerdict.UNDER_COMMIT], label="Under-commit"),
        mpatches.Patch(color=_PLAT_COLORS[PlateauVerdict.MISSING],      label="Missing"),
    ]
    ax2.legend(handles=legend_els2, fontsize=7.5, loc="upper right", ncol=3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gas analysis plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point (standalone smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from parser import LapDataParser
    from brake_analysis import _arc_length   # reuse helper

    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    ref_arc    = _arc_length(ref_states)

    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()
    slow_arc    = _arc_length(slow_states)

    ref_apps  = detect_throttle_applications(ref_states,  ref_arc)
    slow_apps = detect_throttle_applications(slow_states, slow_arc)
    print(f"  Ref throttle applications:  {len(ref_apps)}")
    print(f"  Slow throttle applications: {len(slow_apps)}")

    ref_plats  = detect_throttle_plateaus(ref_states,  ref_arc)
    slow_plats = detect_throttle_plateaus(slow_states, slow_arc)
    print(f"  Ref full-throttle plateaus:  {len(ref_plats)}")
    print(f"  Slow full-throttle plateaus: {len(slow_plats)}")

    app_matches  = match_throttle_applications(ref_apps,  slow_apps)
    plat_matches = match_throttle_plateaus(ref_plats, slow_plats)

    print_gas_recommendations(app_matches, plat_matches)

    save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_gas_analysis(
        ref_states, slow_states, ref_arc, slow_arc,
        app_matches, plat_matches,
        ref_apps, slow_apps,
        ref_plats, slow_plats,
        save_path=save,
    )