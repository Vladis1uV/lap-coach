"""
Spike detection, lap-timing comparison, and matplotlib visualisation.

Workflow
--------
1. Detect "action spikes" (brake / throttle / steer events) in the reference
   (fast) lap using a simple threshold + minimum-gap filter.
2. For each reference spike, search a spatial vicinity in the slow lap to see
   whether the driver performed the same action — earlier, later, or not at all.
3. Emit human-readable coaching recommendations.
4. Plot overlaid channel traces (brake, gas, steering, speed) aligned on
   arc-length along the track, with spike annotations.

Dependencies
------------
    pip install matplotlib numpy scipy
    (lap_data module must be on PYTHONPATH)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import find_peaks

from parser import CarState, LapDataParser, match_laps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arc_length(states: list[CarState]) -> np.ndarray:
    """Cumulative 2-D arc-length (metres) along the list of CarState positions."""
    s = np.zeros(len(states))
    for i in range(1, len(states)):
        dx = states[i].x - states[i - 1].x
        dy = states[i].y - states[i - 1].y
        s[i] = s[i - 1] + math.hypot(dx, dy)
    return s


def _channel(states: list[CarState], attr: str) -> np.ndarray:
    return np.array([getattr(st, attr) for st in states])


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

@dataclass
class Spike:
    """A single detected action spike in one channel of one lap."""
    channel: str          # "brake" | "gas" | "steering"
    index: int            # index into the states list
    arc_pos: float        # arc-length position (m)
    value: float          # peak value
    x: float
    y: float


def detect_spikes(
    states: list[CarState],
    arc: np.ndarray,
    channel: str,
    *,
    height: float = 0.15,          # minimum peak height (normalised 0-1 or rad)
    prominence: float = 0.10,      # scipy prominence threshold
    min_gap_m: float = 20.0,       # minimum metres between two spikes
) -> list[Spike]:
    """Return spikes in *channel* for *states* using scipy.find_peaks.

    Parameters
    ----------
    height:      Absolute minimum value to count as a spike.
    prominence:  How much the peak must stand out from its surroundings.
    min_gap_m:   Minimum arc-length distance (m) between consecutive spikes.
                 Converts to a sample-based `distance` for find_peaks.
    """
    signal = _channel(states, channel)

    # For steering we care about magnitude (left and right turns)
    if channel == "steering":
        signal = np.abs(signal)

    # Estimate average sample spacing to convert metres → samples
    if len(arc) > 1:
        avg_spacing = (arc[-1] - arc[0]) / (len(arc) - 1)
        min_gap_samples = max(1, int(min_gap_m / avg_spacing)) if avg_spacing > 0 else 1
    else:
        min_gap_samples = 1

    peak_indices, props = find_peaks(
        signal,
        height=height,
        prominence=prominence,
        distance=min_gap_samples,
    )

    return [
        Spike(
            channel=channel,
            index=int(idx),
            arc_pos=float(arc[idx]),
            value=float(signal[idx]),
            x=states[idx].x,
            y=states[idx].y,
        )
        for idx in peak_indices
    ]


# ---------------------------------------------------------------------------
# Spike matching & recommendations
# ---------------------------------------------------------------------------

class Verdict(Enum):
    ON_TIME   = auto()   # slow lap has a matching spike within tolerance
    TOO_LATE  = auto()   # slow lap spike is after the reference
    TOO_EARLY = auto()   # slow lap spike is before the reference
    MISSING   = auto()   # no spike found in search window


@dataclass
class SpikeMatch:
    ref_spike: Spike
    slow_spike: Spike | None       # None when MISSING
    verdict: Verdict
    offset_m: float                # slow_arc − ref_arc (+ = later on track)
    recommendation: str


def _find_nearest_spike(
    target_arc: float,
    slow_spikes: list[Spike],
    search_radius_m: float,
) -> Spike | None:
    """Return the slow spike closest to *target_arc* within *search_radius_m*, or None."""
    candidates = [
        sp for sp in slow_spikes
        if abs(sp.arc_pos - target_arc) <= search_radius_m
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda sp: abs(sp.arc_pos - target_arc))


_CHANNEL_VERBS: dict[str, tuple[str, str]] = {
    # channel → (action_word, earlier_advice)
    "brake":    ("brake",  "brake earlier"),
    "gas":      ("apply throttle", "get on the gas sooner"),
    "steering": ("turn",   "initiate the turn earlier"),
}

_LATE_ADVICE: dict[str, str] = {
    "brake":    "brake later / carry more speed",
    "gas":      "hold off the throttle a bit longer",
    "steering": "delay your turn-in",
}


def match_spikes(
    ref_spikes: list[Spike],
    slow_spikes: list[Spike],
    *,
    on_time_window_m: float = 10.0,   # ±m considered "on time"
    search_radius_m: float = 60.0,    # wider window to detect shifted spikes
) -> list[SpikeMatch]:
    """
    For every reference spike, decide whether the slow lap driver did the same
    action on time, too early, too late, or not at all.
    """
    results: list[SpikeMatch] = []
    verb, earlier = _CHANNEL_VERBS.get(ref_spikes[0].channel if ref_spikes else "brake",
                                       ("act", "act earlier"))

    for ref in ref_spikes:
        ch = ref.channel
        verb, earlier_adv = _CHANNEL_VERBS.get(ch, ("act", "act earlier"))
        late_adv = _LATE_ADVICE.get(ch, "act later")

        slow = _find_nearest_spike(ref.arc_pos, slow_spikes, search_radius_m)

        if slow is None:
            verdict = Verdict.MISSING
            offset = float("nan")
            rec = (
                f"At ~{ref.arc_pos:.0f} m you should {verb} "
                f"(peak ≈ {ref.value:.2f}) — no matching action detected."
            )
        else:
            offset = slow.arc_pos - ref.arc_pos
            if abs(offset) <= on_time_window_m:
                verdict = Verdict.ON_TIME
                rec = (
                    f"At ~{ref.arc_pos:.0f} m your {ch} timing is good "
                    f"(offset {offset:+.1f} m)."
                )
            elif offset > 0:
                # slow lap spike is further along the track → driver acts later
                verdict = Verdict.TOO_LATE
                rec = (
                    f"At ~{ref.arc_pos:.0f} m you should {earlier_adv} — "
                    f"you were {abs(offset):.1f} m late "
                    f"(your peak at {slow.arc_pos:.0f} m, "
                    f"reference at {ref.arc_pos:.0f} m)."
                )
            else:
                verdict = Verdict.TOO_EARLY
                rec = (
                    f"At ~{ref.arc_pos:.0f} m you can {late_adv} — "
                    f"you acted {abs(offset):.1f} m too early "
                    f"(your peak at {slow.arc_pos:.0f} m, "
                    f"reference at {ref.arc_pos:.0f} m)."
                )

        results.append(SpikeMatch(
            ref_spike=ref,
            slow_spike=slow,
            verdict=verdict,
            offset_m=offset,
            recommendation=rec,
        ))

    return results


# ---------------------------------------------------------------------------
# Full analysis across all channels
# ---------------------------------------------------------------------------

@dataclass
class LapAnalysis:
    ref_states:  list[CarState]
    slow_states: list[CarState]
    ref_arc:     np.ndarray
    slow_arc:    np.ndarray

    # Spikes per channel
    ref_spikes:  dict[str, list[Spike]]  = field(default_factory=dict)
    slow_spikes: dict[str, list[Spike]]  = field(default_factory=dict)

    # Matched results per channel
    matches: dict[str, list[SpikeMatch]] = field(default_factory=dict)


CHANNELS = ["brake", "gas", "steering"]

SPIKE_PARAMS: dict[str, dict] = {
    "brake":    dict(height=0.10, prominence=0.08, min_gap_m=15.0),
    "gas":      dict(height=0.20, prominence=0.15, min_gap_m=20.0),
    "steering": dict(height=0.05, prominence=0.04, min_gap_m=15.0),
}


def analyse_laps(
    ref_states: list[CarState],
    slow_states: list[CarState],
    *,
    on_time_window_m: float = 10.0,
    search_radius_m: float = 60.0,
) -> LapAnalysis:
    """Run the full spike-detection and matching pipeline."""
    ref_arc  = _arc_length(ref_states)
    slow_arc = _arc_length(slow_states)

    result = LapAnalysis(
        ref_states=ref_states,
        slow_states=slow_states,
        ref_arc=ref_arc,
        slow_arc=slow_arc,
    )

    for ch in CHANNELS:
        params = SPIKE_PARAMS[ch]
        result.ref_spikes[ch]  = detect_spikes(ref_states,  ref_arc,  ch, **params)
        result.slow_spikes[ch] = detect_spikes(slow_states, slow_arc, ch, **params)

        if result.ref_spikes[ch]:
            result.matches[ch] = match_spikes(
                result.ref_spikes[ch],
                result.slow_spikes[ch],
                on_time_window_m=on_time_window_m,
                search_radius_m=search_radius_m,
            )
        else:
            result.matches[ch] = []

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_recommendations(analysis: LapAnalysis) -> None:
    """Print coaching recommendations grouped by channel to stdout."""
    verdict_order = [Verdict.MISSING, Verdict.TOO_LATE, Verdict.TOO_EARLY, Verdict.ON_TIME]
    verdict_label = {
        Verdict.MISSING:   "MISSING",
        Verdict.TOO_LATE:  "TOO LATE",
        Verdict.TOO_EARLY: "TOO EARLY",
        Verdict.ON_TIME:   "ON TIME",
    }

    for ch in CHANNELS:
        matches = analysis.matches.get(ch, [])
        if not matches:
            continue
        print(f"\n{'='*60}")
        print(f"  {ch.upper()} — {len(matches)} reference spikes")
        print(f"{'='*60}")
        for m in sorted(matches, key=lambda m: verdict_order.index(m.verdict)):
            print(f"  [{verdict_label[m.verdict]}]  {m.recommendation}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_CHANNEL_LABELS = {
    "brake":    ("Brake input", "0–1 (normalised)"),
    "gas":      ("Throttle input", "0–1 (normalised)"),
    "steering": ("Steering angle", "rad"),
    "speed":    ("Speed", "m/s"),
}

_VERDICT_COLORS = {
    Verdict.MISSING:   "#e74c3c",
    Verdict.TOO_LATE:  "#e67e22",
    Verdict.TOO_EARLY: "#f1c40f",
    Verdict.ON_TIME:   "#2ecc71",
}


def plot_analysis(
    analysis: LapAnalysis,
    channels_to_plot: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot overlaid reference vs slow-lap traces for each channel.

    Each subplot shows:
      - Reference (fast) lap in blue
      - Slow lap in orange (re-indexed on its own arc-length)
      - Reference spike markers (▼, colour-coded by verdict)
      - Slow-lap spike markers (▲)
      - Shaded search windows around reference spikes

    Parameters
    ----------
    channels_to_plot : list of channel names to include; defaults to all + speed.
    save_path        : if given, save figure to this path instead of showing it.
    """
    if channels_to_plot is None:
        channels_to_plot = ["brake", "gas", "steering", "speed"]

    n = len(channels_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Lap Comparison — Reference (blue) vs Slow Lap (orange)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    ref_arc  = analysis.ref_arc
    slow_arc = analysis.slow_arc

    for ax, ch in zip(axes, channels_to_plot):
        ref_signal  = _channel(analysis.ref_states,  ch)
        slow_signal = _channel(analysis.slow_states, ch)

        # -- traces --
        ax.plot(ref_arc,  ref_signal,  color="#2980b9", lw=1.4,
                label="Reference (fast)", zorder=3)
        ax.plot(slow_arc, slow_signal, color="#e67e22", lw=1.4,
                alpha=0.85, label="Slow lap", zorder=3)

        title, ylabel = _CHANNEL_LABELS.get(ch, (ch.capitalize(), ""))
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Arc-length along track (m)", fontsize=9)
        ax.grid(True, alpha=0.3)

        if ch not in CHANNELS:
            # speed has no spike analysis; just draw the traces
            ax.legend(fontsize=8)
            continue

        # -- reference spike markers (▼) --
        ref_spikes  = analysis.ref_spikes.get(ch, [])
        slow_spikes = analysis.slow_spikes.get(ch, [])
        matches     = {m.ref_spike.index: m for m in analysis.matches.get(ch, [])}

        # Shaded search windows first (behind everything)
        search_radius = 60.0  # should match analyse_laps parameter
        for sp in ref_spikes:
            m = matches.get(sp.index)
            color = _VERDICT_COLORS.get(m.verdict if m else Verdict.MISSING, "#aaa")
            ax.axvspan(
                sp.arc_pos - search_radius,
                sp.arc_pos + search_radius,
                alpha=0.07, color=color, zorder=1,
            )

        # Reference spikes ▼
        for sp in ref_spikes:
            m = matches.get(sp.index)
            color = _VERDICT_COLORS.get(m.verdict if m else Verdict.MISSING, "#aaa")
            ax.annotate(
                "",
                xy=(sp.arc_pos, sp.value),
                xytext=(sp.arc_pos, sp.value + 0.12 * (ax.get_ylim()[1] - ax.get_ylim()[0] or 1)),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                zorder=5,
            )
            ax.plot(sp.arc_pos, sp.value, "v", color=color, ms=8, zorder=6)

        # Slow spike markers ▲ (plain, no colour coding)
        for sp in slow_spikes:
            sig_val = abs(sp.value) if ch == "steering" else sp.value
            ax.plot(sp.arc_pos, sig_val, "^", color="#e67e22", ms=6,
                    alpha=0.9, zorder=6)

        # Legend
        legend_elements = [
            mpatches.Patch(color="#2980b9", label="Reference (fast)"),
            mpatches.Patch(color="#e67e22", label="Slow lap"),
            plt.Line2D([0], [0], marker="v", color="w",
                       markerfacecolor="#555", ms=9, label="Ref spike ▼ (colour = verdict)"),
            plt.Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="#e67e22", ms=8, label="Slow spike ▲"),
            mpatches.Patch(color=_VERDICT_COLORS[Verdict.ON_TIME],   alpha=0.4, label="On time"),
            mpatches.Patch(color=_VERDICT_COLORS[Verdict.TOO_LATE],  alpha=0.4, label="Too late"),
            mpatches.Patch(color=_VERDICT_COLORS[Verdict.TOO_EARLY], alpha=0.4, label="Too early"),
            mpatches.Patch(color=_VERDICT_COLORS[Verdict.MISSING],   alpha=0.4, label="Missing"),
        ]
        ax.legend(handles=legend_elements, fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading reference (fast) lap …")
    ref_parser = LapDataParser(source_mcap="data/hackathon/hackathon_fast_laps.mcap")
    ref_states = ref_parser.get_lap_data()
    print(f"  {len(ref_states)} samples")

    print("Loading slow (good) lap …")
    slow_parser = LapDataParser(source_mcap="data/hackathon/hackathon_good_lap.mcap")
    slow_states = slow_parser.get_lap_data()
    print(f"  {len(slow_states)} samples")

    print("Analysing …")
    analysis = analyse_laps(ref_states, slow_states)

    print_recommendations(analysis)

    save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_analysis(analysis, save_path=save)