"""parser.py"""
import math
import pathlib
from dataclasses import dataclass

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

    @classmethod
    def from_state_estimation(cls, msg, timestamp_ns: int) -> "CarState":
        vx = getattr(msg, "vx_mps", 0.0)
        vy = getattr(msg, "vy_mps", 0.0)
        return cls(
            timestamp=timestamp_ns * 1e-9,
            x=msg.x_m,
            y=msg.y_m,
            z=msg.z_m,
            steering=msg.delta_wheel_rad,
            brake=msg.brake,
            gas=msg.gas,
            speed=math.hypot(vx, vy),
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
# Arc-length for the REFERENCE lap only
# ---------------------------------------------------------------------------

def _arc_length(states: list[CarState]) -> np.ndarray:
    """Cumulative 2-D arc-length (m) for a lap, starting at 0.

    Only call this on the REFERENCE (fast) lap.
    For the slow lap, use map_slow_to_ref() instead so its x-axis
    is expressed purely as reference-point indices with no arc calculation.
    """
    n = len(states)
    arc = np.zeros(n, dtype=float)
    for i in range(1, n):
        arc[i] = arc[i - 1] + math.hypot(
            states[i].x - states[i - 1].x,
            states[i].y - states[i - 1].y,
        )
    return arc


# ---------------------------------------------------------------------------
# Slow-lap → reference mapping  (no arc-length calculation)
# ---------------------------------------------------------------------------

def map_slow_to_ref(
    slow_states: list[CarState],
    ref_states:  list[CarState],
) -> np.ndarray:
    """For every slow-lap point return the index of the nearest reference point.

    The returned array is the x-axis for ALL slow-lap plots and event
    detection.  It contains reference-point indices, not metres — no
    arc-length is computed for the slow lap at any stage.

    Shape: (len(slow_states),) dtype int
    """
    ref_xy  = np.column_stack([[s.x for s in ref_states],  [s.y for s in ref_states]])
    slow_xy = np.column_stack([[s.x for s in slow_states], [s.y for s in slow_states]])
    tree = KDTree(ref_xy)
    _, indices = tree.query(slow_xy)
    return indices.astype(int)


def slow_x_axis(
    slow_to_ref: np.ndarray,
    ref_arc:     np.ndarray,
) -> np.ndarray:
    """Convert the slow-lap index map to reference arc-length values.

    Use this only for PLOTTING so the x-axis has readable metre labels.
    Event detection (detect_brake_events etc.) should receive ref_arc[slow_to_ref]
    directly — it is still derived from the mapping, never from slow-lap XY.

    slow_x[i] = ref_arc[ slow_to_ref[i] ]
    """
    return ref_arc[slow_to_ref]


def align_laps(
    ref_states:  list[CarState],
    slow_states: list[CarState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-stop setup for a ref / slow comparison.

    Returns
    -------
    ref_arc      : arc-length for the reference lap (m), starting at 0
    slow_arc     : slow-lap x-axis expressed in reference metres
                   = ref_arc[ slow_to_ref[i] ]  — no independent calculation
    slow_to_ref  : integer index array, slow_to_ref[i] is the reference index
                   nearest to slow_states[i]
    """
    ref_arc    = _arc_length(ref_states)
    slow_to_ref = map_slow_to_ref(slow_states, ref_states)
    slow_arc   = slow_x_axis(slow_to_ref, ref_arc)
    return ref_arc, slow_arc, slow_to_ref


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
# Delta helpers
# ---------------------------------------------------------------------------

@dataclass
class PositionDelta:
    source_idx: int
    x: float
    y: float
    delta_gas: float
    delta_brake: float
    delta_steering: float
    delta_speed: float


def compute_deltas(
    pairs: list[tuple[CarState, CarState]],
    threshold_gas: float = 0.05,
    threshold_brake: float = 0.05,
    threshold_speed: float = 1.0,
) -> list[PositionDelta]:
    deltas: list[PositionDelta] = []
    for i, (ours, best) in enumerate(pairs):
        d_gas   = best.gas      - ours.gas
        d_brake = best.brake    - ours.brake
        d_steer = best.steering - ours.steering
        d_speed = best.speed    - ours.speed
        if (abs(d_gas) > threshold_gas
                or abs(d_brake) > threshold_brake
                or abs(d_speed) > threshold_speed):
            deltas.append(PositionDelta(
                source_idx=i, x=ours.x, y=ours.y,
                delta_gas=d_gas, delta_brake=d_brake,
                delta_steering=d_steer, delta_speed=d_speed,
            ))
    return deltas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fast_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    good_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()

    ref_arc, slow_arc, slow_to_ref = align_laps(fast_states, good_states)

    print(f"Fast samples : {len(fast_states)},  ref_arc 0 → {ref_arc[-1]:.1f} m")
    print(f"Good samples : {len(good_states)}")
    print(f"slow_arc (= ref positions): min={slow_arc.min():.1f}  max={slow_arc.max():.1f} m")
    print(f"No arc-length was computed for the slow lap.")