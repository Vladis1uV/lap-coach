"""parser.py"""
import math
import pathlib
from dataclasses import dataclass, replace

from mcap_ros2.reader import read_ros2_messages
import numpy as np
from scipy.spatial import KDTree

DIR = pathlib.Path(__file__).parent.parent.parent

TOPICS = ["/constructor0/state_estimation"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CarState:
    timestamp: float
    x: float
    y: float
    z: float
    steering: float   # delta_wheel_rad
    brake: float
    gas: float
    speed: float      # hypot(vx, vy)
    arc: float        # track-aligned arc-length s (m)

    @classmethod
    def from_state_estimation(cls, msg, timestamp_ns: int) -> "CarState":
        vx = getattr(msg, "vx_mps", 0.0)
        vy = getattr(msg, "vy_mps", 0.0)

        idx = msg.sn_map_state.track_sn_state.sn_state.idx
        ds = msg.sn_map_state.track_sn_state.sn_state.ds

        return cls(
            timestamp=timestamp_ns * 1e-9,
            x=msg.x_m,
            y=msg.y_m,
            z=msg.z_m,
            steering=msg.delta_wheel_rad,
            brake=float(msg.brake) / float(5e6),
            gas=msg.gas,
            speed=math.hypot(vx, vy),
            arc=float(idx)+ds,
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class LapDataParser:
    def __init__(
        self,
        source_mcap: str = r"data/hackathon/hackathon_fast_laps.mcap",
        topics: list[str] | None = None,
    ) -> None:
        clean = source_mcap.replace("\\", "/")
        self.mcap_path: pathlib.Path = DIR / clean
        self.topics: list[str] = topics if topics is not None else TOPICS

    def get_lap_data(self) -> list[CarState]:
        states: list[CarState] = []
        for msg_view in read_ros2_messages(self.mcap_path, self.topics):
            states.append(CarState.from_state_estimation(
                msg_view.ros_msg,
                msg_view.log_time_ns,
            ))
        return states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_list(xs: list[CarState], start_idx: int) -> list[CarState]:
    """Cyclically rotate a list so xs[start_idx] becomes the first element."""
    if not xs:
        return []
    start_idx %= len(xs)
    return xs[start_idx:] + xs[:start_idx]


def _lap_length_from_arc(states: list[CarState]) -> float:
    """
    Estimate total lap length from the stored arc field.

    Assumes `state.arc` is a lap-progress coordinate on one lap, typically in
    [0, lap_length).  We use the maximum observed arc value as the lap length
    proxy. If your data contains an explicit total lap length, use that instead.
    """
    if not states:
        return 0.0
    return max(float(s.arc) for s in states)


def _normalize_arc_values(
    states: list[CarState],
    start_arc: float,
    lap_length: float,
) -> np.ndarray:
    """
    Shift arc-lengths so `start_arc` becomes 0, wrapping cyclically by lap length.

    new_arc[i] = (states[i].arc - start_arc) mod lap_length
    """
    if not states:
        return np.array([], dtype=float)

    if lap_length <= 0.0:
        return np.zeros(len(states), dtype=float)

    arc = np.array([float(s.arc) for s in states], dtype=float)
    out = (arc - start_arc) % lap_length

    # Make sure the chosen start really becomes 0 numerically
    if len(out) > 0:
        out[0] = 0.0

    return out


def _states_with_replaced_arc(
    states: list[CarState],
    new_arc: np.ndarray,
) -> list[CarState]:
    """Return copies of states with `arc` replaced by values from new_arc."""
    return [replace(s, arc=float(a)) for s, a in zip(states, new_arc)]


def _closest_state_index(
    query_state: CarState,
    candidates: list[CarState],
) -> int:
    """Index of the candidate closest in XY to `query_state`."""
    if not candidates:
        raise ValueError("Cannot find closest state in an empty list.")

    cand_xy = np.column_stack([[s.x for s in candidates], [s.y for s in candidates]])
    tree = KDTree(cand_xy)
    _, idx = tree.query([query_state.x, query_state.y])
    return int(idx)


# ---------------------------------------------------------------------------
# Alignment using stored arc-length
# ---------------------------------------------------------------------------

def align_laps(
    ref_states: list[CarState],
    slow_states: list[CarState],
) -> tuple[list[CarState], list[CarState]]:
    """
    Align laps using the stored `arc` field inside each CarState.

    Procedure
    ---------
    1. Treat ref_states[0] as the reference origin.
    2. Normalize reference arcs so ref_states_aligned[0].arc == 0.
    3. Find the slow-lap sample closest in XY to ref_states[0].
    4. Rotate slow_states so that sample becomes index 0.
    5. Normalize rotated slow arcs so slow_states_aligned[0].arc == 0.

    Returns
    -------
    ref_states_aligned : list[CarState]
        Reference states with arc shifted so the first state has arc 0.

    slow_states_aligned : list[CarState]
        Rotated slow states with arc shifted so the first state has arc 0.
    """
    if not ref_states:
        raise ValueError("ref_states must not be empty.")
    if not slow_states:
        raise ValueError("slow_states must not be empty.")

    ref_len = _lap_length_from_arc(ref_states)
    slow_len = _lap_length_from_arc(slow_states)

    # Normalize reference lap so its first sample is at arc = 0
    ref_start_arc = float(ref_states[0].arc)
    ref_arc_norm = _normalize_arc_values(ref_states, ref_start_arc, ref_len)
    ref_states_aligned = _states_with_replaced_arc(ref_states, ref_arc_norm)

    # Find slow sample closest to reference start, rotate slow lap
    slow_start_idx = _closest_state_index(ref_states[0], slow_states)
    slow_states_rot = _rotate_list(slow_states, slow_start_idx)

    # Normalize rotated slow lap so its first sample is at arc = 0
    slow_start_arc = float(slow_states_rot[0].arc)
    slow_arc_norm = _normalize_arc_values(slow_states_rot, slow_start_arc, slow_len)
    slow_states_aligned = _states_with_replaced_arc(slow_states_rot, slow_arc_norm)

    return ref_states_aligned, slow_states_aligned


def filter_arc_jumps(
    states: list[CarState],
    *,
    max_jump_m: float = 20.0,
) -> list[CarState]:
    """
    Remove states whose arc value jumps implausibly relative to the previous kept state.

    Assumptions
    -----------
    - `states` are already aligned, rotated, and arc-normalized
    - along a valid lap, consecutive arc values should change smoothly
    - occasional corrupted samples may have arc values from a completely different
      part of the track; these are discarded

    Parameters
    ----------
    states : list[CarState]
        Input states in temporal order.
    max_jump_m : float
        Maximum allowed absolute arc jump between consecutive kept samples.
        If abs(curr.arc - prev_kept.arc) > max_jump_m, the sample is skipped.

    Returns
    -------
    list[CarState]
        Filtered list with corrupted arc-jump samples removed.
    """
    if not states:
        return []

    filtered = [states[0]]
    prev = states[0]

    for s in states[1:]:
        jump = abs(float(s.arc) - float(prev.arc))
        if jump <= max_jump_m:
            filtered.append(s)
            prev = s
        # else: skip corrupted point

    return filtered


# ---------------------------------------------------------------------------
# Matching utilities (kept for backward compat)
# ---------------------------------------------------------------------------

def _euclidean_dist_2d(a: CarState, b: CarState) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def match_laps(
    source: list[CarState],
    target: list[CarState],
) -> dict[int, int]:
    """source_index -> nearest target_index via KDTree."""
    tree = KDTree(np.column_stack([[s.x for s in target], [s.y for s in target]]))
    src_xy = np.column_stack([[s.x for s in source], [s.y for s in source]])
    _, indices = tree.query(src_xy)
    return {i: int(idx) for i, idx in enumerate(indices)}


def match_lap_states(
    source: list[CarState],
    target: list[CarState],
) -> list[tuple[CarState, CarState]]:
    index_map = match_laps(source, target)
    return [(source[i], target[j]) for i, j in index_map.items()]



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fast_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    good_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()

    ref_arc, slow_arc, fast_states_aligned, good_states_aligned = align_laps(
        fast_states,
        good_states,
    )

    print(f"Fast samples : {len(fast_states_aligned)}, ref_arc  0 → {ref_arc[-1]:.1f} m")
    print(f"Good samples : {len(good_states_aligned)}, slow_arc 0 → {slow_arc[-1]:.1f} m")
    print(f"Both laps now start at 0 m.")