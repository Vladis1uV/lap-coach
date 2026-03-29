"""
steering_analysis.py
====================

Steering comparison between a reference lap and a slower lap.

Logic
-----
For each slow_state:
  - find the closest reference state geometrically
  - compute steering delta:

        delta = ref_state.steering - slow_state.steering

  - if |delta| <= threshold, treat as matched
  - otherwise classify the needed correction

Consecutive offsets with the same driving meaning are grouped into regions and
reported like:

    "From 120 m to 145 m you should be turning to the right more sharply."

Important nuance
----------------
The same sign of delta can mean different advice depending on the current turn
direction:

- positive delta:
    * if currently turning right  -> turn right more sharply
    * if currently turning left   -> turn left more smoothly

- negative delta:
    * if currently turning right  -> turn right more smoothly
    * if currently turning left   -> turn left more sharply

If steering is near zero, we fall back to:
    * positive delta -> add more right steering
    * negative delta -> add more left steering
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from parser import LapDataParser, align_laps

try:
    from parser import filter_arc_jumps
except ImportError:
    filter_arc_jumps = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class SteeringVerdict(Enum):
    MATCHED = auto()
    RIGHT_SHARPER = auto()
    RIGHT_SMOOTHER = auto()
    LEFT_SHARPER = auto()
    LEFT_SMOOTHER = auto()
    ADD_RIGHT = auto()
    ADD_LEFT = auto()


@dataclass
class SteeringOffsetPoint:
    slow_idx: int
    ref_idx: int
    arc: float
    x: float
    y: float
    slow_steering: float
    ref_steering: float
    delta: float              # ref - slow
    verdict: SteeringVerdict


@dataclass
class SteeringRecommendation:
    verdict: SteeringVerdict
    start_idx: int
    end_idx: int
    start_arc: float
    end_arc: float
    mean_delta: float
    max_abs_delta: float
    recommendation: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xy_sq(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _classify_steering_delta(
    slow_steering: float,
    ref_steering: float,
    delta_threshold: float,
    neutral_eps: float = 0.002,
) -> SteeringVerdict:
    """
    Classify steering correction.

    delta = ref - slow

    Positive delta means "increase steering value".
    Negative delta means "decrease steering value".

    The wording depends on current turning direction.
    """
    delta = ref_steering - slow_steering

    if abs(delta) <= delta_threshold:
        return SteeringVerdict.MATCHED

    turn_sign = slow_steering
    if abs(turn_sign) < neutral_eps:
        turn_sign = ref_steering

    if abs(turn_sign) < neutral_eps:
        return SteeringVerdict.ADD_RIGHT if delta > 0 else SteeringVerdict.ADD_LEFT

    if turn_sign > 0:
        return SteeringVerdict.RIGHT_SHARPER if delta > 0 else SteeringVerdict.RIGHT_SMOOTHER
    else:
        return SteeringVerdict.LEFT_SMOOTHER if delta > 0 else SteeringVerdict.LEFT_SHARPER


def _verdict_text(verdict: SteeringVerdict) -> str:
    return {
        SteeringVerdict.RIGHT_SHARPER: "turning to the right more sharply",
        SteeringVerdict.RIGHT_SMOOTHER: "turning to the right more smoothly",
        SteeringVerdict.LEFT_SHARPER: "turning to the left more sharply",
        SteeringVerdict.LEFT_SMOOTHER: "turning to the left more smoothly",
        SteeringVerdict.ADD_RIGHT: "adding more right steering",
        SteeringVerdict.ADD_LEFT: "adding more left steering",
        SteeringVerdict.MATCHED: "matching the reference steering",
    }[verdict]


# ---------------------------------------------------------------------------
# Mapping slow -> reference using sliding local search
# ---------------------------------------------------------------------------

def map_slow_to_ref_sliding(
    slow_states: list,
    ref_states: list,
    *,
    initial_window: int = 80,
    step_window: int = 120,
) -> np.ndarray:
    """
    For each slow state, find the closest geometric reference state.

    Uses a sliding local search:
    - first point searches a wider prefix
    - each next point searches around the previous best match

    Assumes both lists are already aligned and sorted by arc.
    """
    if not slow_states or not ref_states:
        return np.array([], dtype=int)

    out = np.zeros(len(slow_states), dtype=int)

    first_hi = min(len(ref_states), initial_window)
    best0 = min(range(first_hi), key=lambda j: _xy_sq(slow_states[0], ref_states[j]))
    out[0] = best0

    for i in range(1, len(slow_states)):
        prev = int(out[i - 1])
        lo = max(0, prev - step_window)
        hi = min(len(ref_states), prev + step_window + 1)

        best = min(range(lo, hi), key=lambda j: _xy_sq(slow_states[i], ref_states[j]))
        out[i] = best

    return out


# ---------------------------------------------------------------------------
# Pointwise steering offsets
# ---------------------------------------------------------------------------

def detect_steering_offsets(
    ref_states: list,
    slow_states: list,
    *,
    delta_threshold: float = 0.008,
    initial_window: int = 80,
    step_window: int = 120,
) -> tuple[list[SteeringOffsetPoint], np.ndarray]:
    """
    Compute significant steering offsets at slow-lap samples.

    Returns
    -------
    offsets : list[SteeringOffsetPoint]
        Only points whose steering mismatch exceeds threshold.
    slow_to_ref : np.ndarray
        The reference index matched to each slow state.
    """
    slow_to_ref = map_slow_to_ref_sliding(
        slow_states,
        ref_states,
        initial_window=initial_window,
        step_window=step_window,
    )

    offsets: list[SteeringOffsetPoint] = []

    for i, slow in enumerate(slow_states):
        ref = ref_states[int(slow_to_ref[i])]
        delta = ref.steering - slow.steering
        verdict = _classify_steering_delta(slow.steering, ref.steering, delta_threshold)

        if verdict == SteeringVerdict.MATCHED:
            continue

        offsets.append(SteeringOffsetPoint(
            slow_idx=i,
            ref_idx=int(slow_to_ref[i]),
            arc=float(slow.arc),
            x=float(slow.x),
            y=float(slow.y),
            slow_steering=float(slow.steering),
            ref_steering=float(ref.steering),
            delta=float(delta),
            verdict=verdict,
        ))

    return offsets, slow_to_ref


# ---------------------------------------------------------------------------
# Grouping consecutive offset points into recommendation regions
# ---------------------------------------------------------------------------

def group_steering_offsets(
    offsets: list[SteeringOffsetPoint],
    *,
    max_gap_m: float = 6.0,
    min_region_m: float = 5.0,
) -> list[SteeringRecommendation]:
    """
    Group consecutive same-verdict offset points into arc regions.
    """
    if not offsets:
        return []

    offsets = sorted(offsets, key=lambda p: p.arc)
    groups: list[list[SteeringOffsetPoint]] = []
    current = [offsets[0]]

    for prev, curr in zip(offsets, offsets[1:]):
        same_kind = curr.verdict == prev.verdict
        close_enough = (curr.arc - prev.arc) <= max_gap_m

        if same_kind and close_enough:
            current.append(curr)
        else:
            groups.append(current)
            current = [curr]

    groups.append(current)

    recs: list[SteeringRecommendation] = []
    for g in groups:
        start_arc = g[0].arc
        end_arc = g[-1].arc
        if end_arc - start_arc < min_region_m:
            continue

        mean_delta = float(np.mean([p.delta for p in g]))
        max_abs_delta = float(np.max(np.abs([p.delta for p in g])))
        verdict = g[0].verdict

        recs.append(SteeringRecommendation(
            verdict=verdict,
            start_idx=g[0].slow_idx,
            end_idx=g[-1].slow_idx,
            start_arc=start_arc,
            end_arc=end_arc,
            mean_delta=mean_delta,
            max_abs_delta=max_abs_delta,
            recommendation=(
                f"Between {start_arc:.0f}–{end_arc:.0f} m you should be "
                f"{_verdict_text(verdict)} "
                f"(average steering offset {mean_delta:+.4f} rad)."
            ),
        ))

    return recs


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_STEER_ICONS = {
    SteeringVerdict.RIGHT_SHARPER: "↗️ ",
    SteeringVerdict.RIGHT_SMOOTHER: "↘️ ",
    SteeringVerdict.LEFT_SHARPER: "↖️ ",
    SteeringVerdict.LEFT_SMOOTHER: "↙️ ",
    SteeringVerdict.ADD_RIGHT: "➡️ ",
    SteeringVerdict.ADD_LEFT: "⬅️ ",
    SteeringVerdict.MATCHED: "✅",
}


def print_steering_recommendations(
    recommendations: list[SteeringRecommendation],
) -> None:
    print("\n" + "=" * 72)
    print("  STEERING ANALYSIS — DRIVER FEEDBACK")
    print("=" * 72)

    if not recommendations:
        print("\n  ✅  Steering matches the reference well — no major issues detected.")
        return

    print("\n--- Steering Recommendations ---")
    for r in recommendations:
        icon = _STEER_ICONS.get(r.verdict, "⚠️ ")
        print(f"  {icon}  {r.recommendation}")

    counts = {}
    for r in recommendations:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    print("\n" + "-" * 72)
    print(f"  🔍  Summary: {len(recommendations)} steering adjustment region(s) detected.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STEER_COLORS = {
    SteeringVerdict.RIGHT_SHARPER: "#e74c3c",
    SteeringVerdict.RIGHT_SMOOTHER: "#f39c12",
    SteeringVerdict.LEFT_SHARPER: "#8e44ad",
    SteeringVerdict.LEFT_SMOOTHER: "#3498db",
    SteeringVerdict.ADD_RIGHT: "#c0392b",
    SteeringVerdict.ADD_LEFT: "#2980b9",
    SteeringVerdict.MATCHED: "#2ecc71",
}


def plot_steering_analysis(
    ref_states: list,
    slow_states: list,
    recommendations: list[SteeringRecommendation],
    slow_to_ref: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Two-panel steering plot.

    Top:
      raw steering traces for ref and slow

    Bottom:
      raw steering traces plus shaded recommendation regions
    """
    ref_arc = np.array([s.arc for s in ref_states], dtype=float)
    slow_arc = np.array([s.arc for s in slow_states], dtype=float)

    ref_steer = np.array([s.steering for s in ref_states], dtype=float)
    slow_steer = np.array([s.steering for s in slow_states], dtype=float)

    matched_ref_steer = np.array(
        [ref_states[int(j)].steering for j in slow_to_ref],
        dtype=float,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    fig.suptitle("Steering Analysis", fontsize=13, fontweight="bold")

    # Top panel
    ax1.set_title("Steering Traces", fontsize=10, loc="left")
    ax1.plot(ref_arc, ref_steer, color="#2980b9", lw=1.4, label="Reference (fast)", zorder=3)
    ax1.plot(slow_arc, slow_steer, color="#e67e22", lw=1.4, alpha=0.9, label="Slow lap", zorder=3)
    ax1.set_ylabel("Steering (rad)", fontsize=9)
    ax1.set_xlabel("Arc-length (m)", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    # Bottom panel
    ax2.set_title("Steering Recommendations", fontsize=10, loc="left")
    ax2.plot(slow_arc, matched_ref_steer, color="#2980b9", lw=1.4, label="Reference at matched points", zorder=3)
    ax2.plot(slow_arc, slow_steer, color="#e67e22", lw=1.4, alpha=0.9, label="Slow lap", zorder=3)
    ax2.set_ylabel("Steering (rad)", fontsize=9)
    ax2.set_xlabel("Arc-length (m)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    for r in recommendations:
        color = _STEER_COLORS[r.verdict]
        ax2.axvspan(r.start_arc, r.end_arc, alpha=0.20, color=color, zorder=1)

        mid = 0.5 * (r.start_arc + r.end_arc)
        short = {
            SteeringVerdict.RIGHT_SHARPER: "right sharper",
            SteeringVerdict.RIGHT_SMOOTHER: "right smoother",
            SteeringVerdict.LEFT_SHARPER: "left sharper",
            SteeringVerdict.LEFT_SMOOTHER: "left smoother",
            SteeringVerdict.ADD_RIGHT: "more right",
            SteeringVerdict.ADD_LEFT: "more left",
        }[r.verdict]

        ax2.text(
            mid,
            np.nanmax([matched_ref_steer.max(), slow_steer.max()]) * 0.90 if max(abs(matched_ref_steer).max(), abs(slow_steer).max()) > 0 else 0.0,
            short,
            ha="center",
            fontsize=7,
            color=color,
            zorder=5,
        )

    legend_els = [
        plt.Line2D([0], [0], color="#2980b9", lw=2, label="Reference at matched points"),
        plt.Line2D([0], [0], color="#e67e22", lw=2, label="Slow lap"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.RIGHT_SHARPER], alpha=0.4, label="Right sharper"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.RIGHT_SMOOTHER], alpha=0.4, label="Right smoother"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.LEFT_SHARPER], alpha=0.4, label="Left sharper"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.LEFT_SMOOTHER], alpha=0.4, label="Left smoother"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.ADD_RIGHT], alpha=0.4, label="Add right"),
        mpatches.Patch(color=_STEER_COLORS[SteeringVerdict.ADD_LEFT], alpha=0.4, label="Add left"),
    ]
    ax2.legend(handles=legend_els, fontsize=7.5, loc="upper right", ncol=3)

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

    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()

    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()

    ref_states, slow_states = align_laps(ref_states, slow_states)

    if filter_arc_jumps is not None:
        ref_states = filter_arc_jumps(ref_states)
        slow_states = filter_arc_jumps(slow_states)

    offsets, slow_to_ref = detect_steering_offsets(
        ref_states,
        slow_states,
        delta_threshold=0.008,
        initial_window=80,
        step_window=120,
    )

    recommendations = group_steering_offsets(
        offsets,
        max_gap_m=6.0,
        min_region_m=5.0,
    )

    print_steering_recommendations(recommendations)

    save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_steering_analysis(
        ref_states,
        slow_states,
        recommendations,
        slow_to_ref,
        save_path=save,
    )