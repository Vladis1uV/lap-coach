"""Lap data parsing and processing logic."""
import math
import pathlib
from dataclasses import dataclass

from mcap_ros2.reader import read_ros2_messages

# Project root — two levels up from this file
DIR = pathlib.Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CarState:
    timestamp: float  # seconds (nanoseconds converted)
    x: float
    y: float
    z: float
    steering: float   # delta_wheel_rad
    brake: float
    gas: float
    speed: float      # derived: magnitude of vx/vy

    @classmethod
    def from_state_estimation(cls, msg, timestamp_ns: int) -> "CarState":
        """Build a CarState from a StateEstimation ROS2 message.

        Field names come from the StateEstimation message definition in
        sd_msgs/.  Adjust if the schema uses different names.
        """
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

TOPICS = ["/constructor0/state_estimation"]


class LapDataParser:
    """Read an MCAP file and expose its state-estimation samples."""

    def __init__(
        self,
        source_mcap: str = r"data/hackathon/hackathon_fast_laps.mcap",
        topics: list[str] | None = None,
    ) -> None:
        # Use forward-slash paths so pathlib works cross-platform.
        # Replace any Windows backslashes just in case.
        clean = source_mcap.replace("\\", "/")
        self.mcap_path: pathlib.Path = DIR / clean
        self.topics: list[str] = topics if topics is not None else TOPICS

    def get_lap_data(self) -> list[CarState]:
        """Read all state-estimation messages and return a list of CarState."""
        states: list[CarState] = []
        for msg_view in read_ros2_messages(self.mcap_path, self.topics):
            state = CarState.from_state_estimation(
                msg_view.ros_msg,
                msg_view.log_time,      # nanoseconds since epoch
            )
            states.append(state)
        return states


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------

def _euclidean_dist_2d(a: CarState, b: CarState) -> float:
    """2-D Euclidean distance between two CarState positions."""
    return math.hypot(a.x - b.x, a.y - b.y)


def match_laps(
    source: list[CarState],
    target: list[CarState],
) -> dict[int, int]:
    """For every point in *source*, return the index of the nearest point in *target*.

    Returns dict: source_index -> target_index.

    Note: naïve O(n·m) search — fine for single-lap lengths (~8 000 pts at
    100 Hz over 80 s).  Switch to a KD-tree (scipy.spatial.KDTree) if speed
    becomes an issue with longer recordings.
    """
    matched: dict[int, int] = {}
    for i, src_pt in enumerate(source):
        closest_idx = min(
            range(len(target)),
            key=lambda j, s=src_pt: _euclidean_dist_2d(s, target[j]),
        )
        matched[i] = closest_idx
    return matched


def match_lap_states(
    source: list[CarState],
    target: list[CarState],
) -> list[tuple[CarState, CarState]]:
    """Pair each point in *source* with the spatially closest point in *target*.

    Returns a list of (source_state, matched_target_state) tuples — one per
    source point — which is easier to iterate than a dict of dicts.
    """
    index_map = match_laps(source, target)
    return [(source[i], target[j]) for i, j in index_map.items()]


# ---------------------------------------------------------------------------
# Delta / recommendation helpers
# ---------------------------------------------------------------------------

@dataclass
class PositionDelta:
    source_idx: int
    x: float
    y: float
    delta_gas: float      # positive → best lap uses more throttle here
    delta_brake: float    # positive → best lap brakes harder here
    delta_steering: float # positive → best lap turns more to the right here
    delta_speed: float    # positive → best lap is faster here


def compute_deltas(
    pairs: list[tuple[CarState, CarState]],
    threshold_gas: float = 0.05,
    threshold_brake: float = 0.05,
    threshold_speed: float = 1.0,   # m/s
) -> list[PositionDelta]:
    """Compute per-position deltas between *our* lap and the *best* lap.

    Only positions where at least one channel exceeds its threshold are kept,
    reducing noise from near-identical driving.

    Args:
        pairs: output of match_lap_states — (our_state, best_state).
        threshold_gas: minimum |Δgas| to include a point.
        threshold_brake: minimum |Δbrake| to include a point.
        threshold_speed: minimum |Δspeed| (m/s) to include a point.

    Returns:
        Filtered list of PositionDelta objects.
    """
    deltas: list[PositionDelta] = []
    for i, (ours, best) in enumerate(pairs):
        d_gas = best.gas - ours.gas
        d_brake = best.brake - ours.brake
        d_steer = best.steering - ours.steering
        d_speed = best.speed - ours.speed

        if (
            abs(d_gas) > threshold_gas
            or abs(d_brake) > threshold_brake
            or abs(d_speed) > threshold_speed
        ):
            deltas.append(
                PositionDelta(
                    source_idx=i,
                    x=ours.x,
                    y=ours.y,
                    delta_gas=d_gas,
                    delta_brake=d_brake,
                    delta_steering=d_steer,
                    delta_speed=d_speed,
                )
            )
    return deltas


# ---------------------------------------------------------------------------
# Entry point / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_parser = LapDataParser(
        source_mcap="data/hackathon/hackathon_fast_laps.mcap"
    )
    our_parser = LapDataParser(
        source_mcap="data/hackathon/hackathon_good_lap.mcap"
    )

    print("Loading fast laps …")
    best_states = best_parser.get_lap_data()
    print(f"  {len(best_states)} samples loaded")

    print("Loading good lap …")
    our_states = our_parser.get_lap_data()
    print(f"  {len(our_states)} samples loaded")

    print("Matching lap positions …")
    pairs = match_lap_states(our_states, best_states)

    print("Computing deltas …")
    deltas = compute_deltas(pairs)
    print(f"  {len(deltas)} significant difference points found\n")

    # --- simple console report ---
    for d in deltas[:20]:   # print first 20 as a quick sanity check
        parts = []
        if abs(d.delta_gas) > 0.05:
            direction = "more" if d.delta_gas > 0 else "less"
            parts.append(f"gas {direction} ({d.delta_gas:+.2f})")
        if abs(d.delta_brake) > 0.05:
            direction = "harder" if d.delta_brake > 0 else "lighter"
            parts.append(f"brake {direction} ({d.delta_brake:+.2f})")
        if abs(d.delta_speed) > 1.0:
            direction = "faster" if d.delta_speed > 0 else "slower"
            parts.append(f"speed {direction} ({d.delta_speed:+.1f} m/s)")
        print(
            f"[{d.source_idx:5d}] pos=({d.x:8.1f}, {d.y:8.1f})  "
            + "  |  ".join(parts)
        )