"""
brake_analysis.py
=================
Brake-event detection and lap comparison.

Brake pressure is a *spike* signal — sharp application, hold, release.
The meaningful events are:

  - **BrakeEvent**: a single braking zone, characterised by:
      · entry_arc  — where pressure crosses a threshold going up (brake point)
      · peak_arc   — maximum pressure (peak braking)
      · exit_arc   — where pressure falls back below threshold (release)
      · peak_value — normalised peak pressure (0–1)
      · duration_m — length of the braking zone in metres

Detection works on a **signal normalised to [0, 1]** within each lap so that
thresholds are scale-independent regardless of whether the raw signal is in
Pascals, bar, or a 0-1 CAN value.

Plots (separate figures / axes):
  1. Brake pressure — reference vs slow, with event markers and search windows
  2. Speed         — reference vs slow (context for braking zones)
  3. Steering      — reference vs slow (shows what driver does mid-brake)

Public API
----------
    detect_brake_events(states, arc, **kwargs) -> list[BrakeEvent]
    match_brake_events(ref, slow, ...)         -> list[BrakeMatch]
    print_brake_recommendations(matches)
    plot_brake_analysis(...)

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
from scipy.signal import find_peaks

from parser import LapDataParser, _arc_length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_channel(states: list, attr: str) -> np.ndarray:
    return np.array([getattr(s, attr) for s in states])


def _smooth(signal: np.ndarray, window: int = 9) -> np.ndarray:
    return uniform_filter1d(signal.astype(float), size=max(1, window))


def _normalise(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (signal / max, max).  max is stored so raw values can be reported."""
    peak = float(signal.max())
    if peak < 1e-9:
        return signal.copy(), 1.0
    return signal / peak, peak


def _arc_to_samples(arc: np.ndarray, metres: float) -> int:
    if len(arc) < 2 or arc[-1] <= arc[0]:
        return 1
    avg = (arc[-1] - arc[0]) / (len(arc) - 1)
    return max(1, int(metres / avg))


# ---------------------------------------------------------------------------
# BrakeEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class BrakeEvent:
    """A single braking zone."""
    # Indices into the states list
    entry_idx: int
    peak_idx:  int
    exit_idx:  int

    # Arc-length positions (m)
    entry_arc:  float
    peak_arc:   float
    exit_arc:   float
    duration_m: float   # exit_arc - entry_arc

    # Normalised (0–1) values
    peak_norm:  float   # peak pressure, normalised

    # Track position
    x_entry: float
    y_entry: float
    x_peak:  float
    y_peak:  float


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_brake_events(
    states: list,
    arc: np.ndarray,
    *,
    # All thresholds operate on the 0-1 normalised signal
    entry_threshold: float  = 0.15,   # normalised pressure to call "braking started"
    peak_min:        float  = 0.35,   # minimum normalised peak to count as an event
    prominence:      float  = 0.25,   # how much the peak must stand out
    min_gap_m:       float  = 40.0,   # minimum metres between two events
    walk_limit_m:    float  = 80.0,   # max metres to walk back/forward from peak
    smooth_window:   int    = 11,     # pre-smoothing (removes sensor buzz)
) -> list[BrakeEvent]:
    """
    Detect braking zones from a (possibly raw-unit) brake signal.

    The signal is normalised to [0, 1] before any threshold is applied, so
    all parameters are scale-independent.

    Algorithm
    ---------
    1. Smooth → normalise → find peaks (scipy).
    2. For each peak walk backward to entry_threshold → entry point.
    3. Walk forward to entry_threshold → exit point.
    4. Enforce min_gap between events.
    """
    raw = _get_channel(states, "brake")
    smoothed = _smooth(raw, smooth_window)
    norm, _scale = _normalise(smoothed)

    min_gap_s  = _arc_to_samples(arc, min_gap_m)
    walk_lim   = _arc_to_samples(arc, walk_limit_m)
    n          = len(norm)

    peak_indices, _ = find_peaks(
        norm,
        height=peak_min,
        prominence=prominence,
        distance=min_gap_s,
    )

    events: list[BrakeEvent] = []

    for pi in peak_indices:
        # Walk backward → entry
        entry_idx = pi
        for k in range(pi - 1, max(0, pi - walk_lim) - 1, -1):
            if norm[k] < entry_threshold:
                entry_idx = k
                break
        else:
            entry_idx = max(0, pi - walk_lim)

        # Walk forward → exit
        exit_idx = pi
        for k in range(pi + 1, min(n, pi + walk_lim)):
            if norm[k] < entry_threshold:
                exit_idx = k
                break
        else:
            exit_idx = min(n - 1, pi + walk_lim)

        events.append(BrakeEvent(
            entry_idx  = int(entry_idx),
            peak_idx   = int(pi),
            exit_idx   = int(exit_idx),
            entry_arc  = float(arc[entry_idx]),
            peak_arc   = float(arc[pi]),
            exit_arc   = float(arc[exit_idx]),
            duration_m = float(arc[exit_idx] - arc[entry_idx]),
            peak_norm  = float(norm[pi]),
            x_entry    = states[entry_idx].x,
            y_entry    = states[entry_idx].y,
            x_peak     = states[pi].x,
            y_peak     = states[pi].y,
        ))

    return events


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class EntryVerdict(Enum):
    ON_TIME   = auto()
    TOO_LATE  = auto()   # slow driver brakes later (carries more speed — or overshot)
    TOO_EARLY = auto()   # slow driver brakes earlier than needed


class PeakVerdict(Enum):
    SIMILAR     = auto()
    TOO_HARD    = auto()   # slow driver peaks higher (panic braking)
    TOO_LIGHT   = auto()   # slow driver peaks lower (under-braking, trail-braking?)


class ReleaseVerdict(Enum):
    ON_TIME   = auto()
    TOO_EARLY = auto()   # slow driver releases pressure before the corner
    TOO_LATE  = auto()   # slow driver holds brakes too long into the corner


# ---------------------------------------------------------------------------
# BrakeMatch
# ---------------------------------------------------------------------------

@dataclass
class BrakeMatch:
    ref:  BrakeEvent
    slow: BrakeEvent | None

    entry_verdict:   EntryVerdict
    peak_verdict:    PeakVerdict
    release_verdict: ReleaseVerdict

    entry_offset_m:   float   # slow.entry_arc  − ref.entry_arc
    peak_offset_m:    float   # slow.peak_arc   − ref.peak_arc
    release_offset_m: float   # slow.exit_arc   − ref.exit_arc
    peak_norm_delta:  float   # slow.peak_norm  − ref.peak_norm

    recommendations: list[str]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _nearest_event(
    target_arc: float,
    candidates: list[BrakeEvent],
    radius_m: float,
) -> BrakeEvent | None:
    within = [e for e in candidates if abs(e.peak_arc - target_arc) <= radius_m]
    return min(within, key=lambda e: abs(e.peak_arc - target_arc)) if within else None


def match_brake_events(
    ref_events:  list[BrakeEvent],
    slow_events: list[BrakeEvent],
    *,
    on_time_window_m:  float = 10.0,
    search_radius_m:   float = 80.0,
    peak_norm_tol:     float = 0.10,   # normalised pressure difference tolerance
) -> list[BrakeMatch]:
    results: list[BrakeMatch] = []

    for ref in ref_events:
        slow = _nearest_event(ref.peak_arc, slow_events, search_radius_m)

        if slow is None:
            results.append(BrakeMatch(
                ref=ref, slow=None,
                entry_verdict=EntryVerdict.TOO_LATE,
                peak_verdict=PeakVerdict.SIMILAR,
                release_verdict=ReleaseVerdict.ON_TIME,
                entry_offset_m=float("nan"),
                peak_offset_m=float("nan"),
                release_offset_m=float("nan"),
                peak_norm_delta=float("nan"),
                recommendations=[
                    f"At ~{ref.peak_arc:.0f} m no braking event found in the slow lap "
                    f"(searched ±{search_radius_m:.0f} m). "
                    f"The reference brakes here to {ref.peak_norm:.0%} — you may be "
                    f"carrying too much speed or missing the braking zone entirely."
                ],
            ))
            continue

        en_off  = slow.entry_arc - ref.entry_arc
        pk_off  = slow.peak_arc  - ref.peak_arc
        rel_off = slow.exit_arc  - ref.exit_arc
        pk_d    = slow.peak_norm - ref.peak_norm

        # Classify entry
        if abs(en_off) <= on_time_window_m:
            en_v = EntryVerdict.ON_TIME
        elif en_off > 0:
            en_v = EntryVerdict.TOO_LATE
        else:
            en_v = EntryVerdict.TOO_EARLY

        # Classify peak
        if abs(pk_d) <= peak_norm_tol:
            pk_v = PeakVerdict.SIMILAR
        elif pk_d > 0:
            pk_v = PeakVerdict.TOO_HARD
        else:
            pk_v = PeakVerdict.TOO_LIGHT

        # Classify release
        if abs(rel_off) <= on_time_window_m:
            rel_v = ReleaseVerdict.ON_TIME
        elif rel_off < 0:
            rel_v = ReleaseVerdict.TOO_EARLY
        else:
            rel_v = ReleaseVerdict.TOO_LATE

        recs: list[str] = []
        z = f"~{ref.peak_arc:.0f} m"

        if en_v == EntryVerdict.TOO_LATE:
            recs.append(
                f"[{z}] Brake {abs(en_off):.0f} m earlier — "
                f"you hit the brakes at {slow.entry_arc:.0f} m, "
                f"reference brakes at {ref.entry_arc:.0f} m. "
                f"Late braking costs you corner entry speed."
            )
        elif en_v == EntryVerdict.TOO_EARLY:
            recs.append(
                f"[{z}] You can brake {abs(en_off):.0f} m later — "
                f"you braked at {slow.entry_arc:.0f} m vs reference {ref.entry_arc:.0f} m. "
                f"Braking earlier than needed scrubs speed unnecessarily."
            )

        if pk_v == PeakVerdict.TOO_HARD:
            recs.append(
                f"[{z}] You're braking harder than the reference "
                f"({slow.peak_norm:.0%} vs {ref.peak_norm:.0%} normalised). "
                f"This often follows a late entry — the extra pressure compensates "
                f"for running out of braking distance."
            )
        elif pk_v == PeakVerdict.TOO_LIGHT:
            recs.append(
                f"[{z}] Your peak brake pressure ({slow.peak_norm:.0%}) is lower than "
                f"the reference ({ref.peak_norm:.0%}). "
                f"Commit to initial braking harder — trail off as you turn in."
            )

        if rel_v == ReleaseVerdict.TOO_EARLY:
            recs.append(
                f"[{z}] You release brakes {abs(rel_off):.0f} m before the reference. "
                f"Releasing too early loses the rotation trail-braking provides — "
                f"hold a little pressure as you turn in."
            )
        elif rel_v == ReleaseVerdict.TOO_LATE:
            recs.append(
                f"[{z}] You hold brakes {abs(rel_off):.0f} m past the reference exit. "
                f"Braking too deep into the corner tightens the radius and "
                f"delays throttle application on exit."
            )

        if not recs:
            recs.append(
                f"[{z}] Braking zone matches the reference well "
                f"(entry Δ{en_off:+.0f} m, peak Δ{pk_d:+.0%}, release Δ{rel_off:+.0f} m)."
            )

        results.append(BrakeMatch(
            ref=ref, slow=slow,
            entry_verdict=en_v,
            peak_verdict=pk_v,
            release_verdict=rel_v,
            entry_offset_m=en_off,
            peak_offset_m=pk_off,
            release_offset_m=rel_off,
            peak_norm_delta=pk_d,
            recommendations=recs,
        ))

    return results


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_ICONS = {
    EntryVerdict.ON_TIME:       "✅",
    EntryVerdict.TOO_LATE:      "⚠️ ",
    EntryVerdict.TOO_EARLY:     "⚠️ ",
    PeakVerdict.SIMILAR:        "✅",
    PeakVerdict.TOO_HARD:       "⚠️ ",
    PeakVerdict.TOO_LIGHT:      "⚠️ ",
    ReleaseVerdict.ON_TIME:     "✅",
    ReleaseVerdict.TOO_EARLY:   "⚠️ ",
    ReleaseVerdict.TOO_LATE:    "⚠️ ",
}


def print_brake_recommendations(matches: list[BrakeMatch]) -> None:
    print("\n" + "=" * 64)
    print(f"  BRAKES — {len(matches)} reference braking zones analysed")
    print("=" * 64)
    for m in matches:
        ei = _ICONS.get(m.entry_verdict,   "?")
        pi = _ICONS.get(m.peak_verdict,    "?")
        ri = _ICONS.get(m.release_verdict, "?")
        print(f"\n  Zone ~{m.ref.peak_arc:.0f} m  |  "
              f"entry {ei}  peak {pi}  release {ri}")
        for rec in m.recommendations:
            print(f"    • {rec}")


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_EN_COL = {
    EntryVerdict.ON_TIME:   "#2ecc71",
    EntryVerdict.TOO_LATE:  "#e74c3c",
    EntryVerdict.TOO_EARLY: "#f1c40f",
}
_PK_COL = {
    PeakVerdict.SIMILAR:   "#2ecc71",
    PeakVerdict.TOO_HARD:  "#e74c3c",
    PeakVerdict.TOO_LIGHT: "#f1c40f",
}
_REL_COL = {
    ReleaseVerdict.ON_TIME:   "#2ecc71",
    ReleaseVerdict.TOO_EARLY: "#f1c40f",
    ReleaseVerdict.TOO_LATE:  "#e74c3c",
}


# ---------------------------------------------------------------------------
# Plotting  (three separate subplots: brake / speed / steering)
# ---------------------------------------------------------------------------

def plot_brake_analysis(
    ref_states:  list,
    slow_states: list,
    ref_arc:     np.ndarray,
    slow_arc:    np.ndarray,
    matches:     list[BrakeMatch],
    ref_events:  list[BrakeEvent],
    slow_events: list[BrakeEvent],
    save_path:   str | None = None,
) -> None:
    """
    Three separate subplots stacked vertically:
      1. Brake pressure (normalised) — event markers + phase lines
      2. Speed (m/s)                 — context for how braking affects speed
      3. Steering angle (rad)        — shows trail-braking / mid-corner inputs
    """
    # Normalise brake signals for display (scale-independent)
    ref_brake_raw  = _get_channel(ref_states,  "brake")
    slow_brake_raw = _get_channel(slow_states, "brake")
    ref_brake_norm,  _ = _normalise(_smooth(ref_brake_raw))
    slow_brake_norm, _ = _normalise(_smooth(slow_brake_raw))

    ref_speed  = _get_channel(ref_states,  "speed")
    slow_speed = _get_channel(slow_states, "speed")
    ref_steer  = _get_channel(ref_states,  "steering")
    slow_steer = _get_channel(slow_states, "steering")

    fig, (ax_brake, ax_speed, ax_steer) = plt.subplots(
        3, 1, figsize=(18, 12), sharex=False
    )
    fig.suptitle(
        "Brake Analysis — Reference (blue) vs Slow Lap (orange)",
        fontsize=13, fontweight="bold",
    )

    # ── 1. Brake pressure ────────────────────────────────────────────────────
    ax_brake.plot(ref_arc,  ref_brake_norm,  color="#2980b9", lw=1.4,
                  label="Reference", zorder=3)
    ax_brake.plot(slow_arc, slow_brake_norm, color="#e67e22", lw=1.4,
                  alpha=0.9, label="Slow lap", zorder=3)
    ax_brake.set_title("Brake pressure (normalised 0–1)", fontsize=10, loc="left")
    ax_brake.set_ylabel("Brake (norm.)")
    ax_brake.set_xlabel("Arc-length (m)")
    ax_brake.set_ylim(-0.05, 1.25)
    ax_brake.grid(True, alpha=0.3)

    # Map ref spike index → match for colour lookup
    ref_idx_to_match = {m.ref.peak_idx: m for m in matches}

    for ev in ref_events:
        m     = ref_idx_to_match.get(ev.peak_idx)
        en_c  = _EN_COL[m.entry_verdict]   if m else "#aaa"
        pk_c  = _PK_COL[m.peak_verdict]    if m else "#aaa"
        rel_c = _REL_COL[m.release_verdict] if m else "#aaa"

        # Shaded braking zone (reference)
        ax_brake.axvspan(ev.entry_arc, ev.exit_arc,
                         alpha=0.10, color="#2980b9", zorder=1)

        # Phase markers: entry (dashed), peak (solid ▼), exit (dotted)
        ax_brake.axvline(ev.entry_arc, color=en_c,  lw=1.3, ls="--", alpha=0.9, zorder=4)
        ax_brake.axvline(ev.peak_arc,  color=pk_c,  lw=1.8, ls="-",  alpha=0.9, zorder=4)
        ax_brake.axvline(ev.exit_arc,  color=rel_c, lw=1.3, ls=":",  alpha=0.9, zorder=4)

        # Peak marker ▼ on the trace
        ax_brake.plot(ev.peak_arc, ev.peak_norm, "v",
                      color=pk_c, ms=9, zorder=6)

        # Offset arrow to slow lap peak (if matched)
        if m and m.slow is not None and abs(m.peak_offset_m) > 3:
            y_arrow = 1.10
            ax_brake.annotate(
                "",
                xy=(m.slow.peak_arc, y_arrow),
                xytext=(ev.peak_arc,  y_arrow),
                arrowprops=dict(arrowstyle="->", color=pk_c, lw=1.5),
                zorder=7,
            )
            mid = (ev.peak_arc + m.slow.peak_arc) / 2
            ax_brake.text(mid, 1.14, f"{m.peak_offset_m:+.0f} m",
                          ha="center", fontsize=7, color=pk_c, zorder=8)

    # Slow-lap event peaks ▲
    for ev in slow_events:
        ax_brake.plot(ev.peak_arc, ev.peak_norm, "^",
                      color="#e67e22", ms=7, alpha=0.85, zorder=5)

    legend_els = [
        mpatches.Patch(color="#2980b9", label="Reference"),
        mpatches.Patch(color="#e67e22", label="Slow lap"),
        plt.Line2D([0], [0], color="#555", ls="--", lw=1.3, label="Entry"),
        plt.Line2D([0], [0], color="#555", ls="-",  lw=1.8, label="Peak"),
        plt.Line2D([0], [0], color="#555", ls=":",  lw=1.3, label="Release"),
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="On time / similar"),
        mpatches.Patch(color="#e74c3c", alpha=0.5, label="Late / too hard"),
        mpatches.Patch(color="#f1c40f", alpha=0.5, label="Early / too light"),
    ]
    ax_brake.legend(handles=legend_els, fontsize=7.5, loc="upper right", ncol=4)

    # ── 2. Speed ─────────────────────────────────────────────────────────────
    ax_speed.plot(ref_arc,  ref_speed,  color="#2980b9", lw=1.4, label="Reference")
    ax_speed.plot(slow_arc, slow_speed, color="#e67e22", lw=1.4, alpha=0.9, label="Slow lap")
    ax_speed.set_title("Speed (m/s)", fontsize=10, loc="left")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_xlabel("Arc-length (m)")
    ax_speed.grid(True, alpha=0.3)

    # Shade braking zones on speed plot too (helps correlate visually)
    for ev in ref_events:
        ax_speed.axvspan(ev.entry_arc, ev.exit_arc, alpha=0.08, color="#2980b9", zorder=1)

    ax_speed.legend(fontsize=8, loc="upper right")

    # ── 3. Steering ──────────────────────────────────────────────────────────
    ax_steer.plot(ref_arc,  ref_steer,  color="#2980b9", lw=1.4, label="Reference")
    ax_steer.plot(slow_arc, slow_steer, color="#e67e22", lw=1.4, alpha=0.9, label="Slow lap")
    ax_steer.axhline(0, color="#bbb", lw=0.7)
    ax_steer.set_title("Steering angle (rad)  — shows trail-braking overlap",
                       fontsize=10, loc="left")
    ax_steer.set_ylabel("Steering (rad)")
    ax_steer.set_xlabel("Arc-length (m)")
    ax_steer.grid(True, alpha=0.3)

    # Shade braking zones so you can see when the driver is steering while braking
    for ev in ref_events:
        ax_steer.axvspan(ev.entry_arc, ev.exit_arc, alpha=0.08, color="#2980b9", zorder=1)

    ax_steer.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Brake analysis plot saved → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    ref_arc    = _arc_length(ref_states)

    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()
    slow_arc    = _arc_length(slow_states)

    print("Detecting brake events …")
    ref_events  = detect_brake_events(ref_states,  ref_arc)
    slow_events = detect_brake_events(slow_states, slow_arc)
    print(f"  Reference brake zones : {len(ref_events)}")
    print(f"  Slow lap brake zones  : {len(slow_events)}")

    print("Matching …")
    matches = match_brake_events(ref_events, slow_events)

    print_brake_recommendations(matches)

    save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_brake_analysis(
        ref_states, slow_states,
        ref_arc, slow_arc,
        matches, ref_events, slow_events,
        save_path=save,
    )