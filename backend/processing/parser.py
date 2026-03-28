"""Lap data parsing and processing logic."""
import math
from typing import Optional
from mcap_ros2.reader import read_ros2_messages
import pathlib

DIR = pathlib.Path(__file__).parent.parent.parent

def parse_lap_data(raw_data: bytes) -> dict:
    """Parse raw lap data and return structured results.

    Replace this with your actual parsing logic.
    """
    # TODO: Implement your data processing here
    text = raw_data.decode("utf-8")
    lines = text.strip().splitlines()

    return {
        "total_lines": len(lines),
        "preview": lines[:5] if lines else [],
    }

def get_lap_data(source_mcap: str = "data\hackathon\hackathon_fast_laps.mcap", topics = ["/constructor0/state_estimation"]) -> dict:
    for msg_view in read_ros2_messages(
        DIR / source_mcap,
        topics,
    ):
        msg = msg_view.ros_msg
        print(msg)
        break

from dataclasses import dataclass

@dataclass
class CarState:
    x: float
    y: float
    z: float
    steering: float
    brake: float
    gas: float

    @classmethod
    def from_state_estimation(cls, msg):
        return cls(
            x=msg.x_m,
            y=msg.y_m,
            z=msg.z_m,
            steering=msg.delta_wheel_rad,
            brake=msg.brake,
            gas=msg.gas,
        )

class LapDataParser:
    def __init__(self, source_mcap: str = "data\hackathon\hackathon_fast_laps.mcap", topics = ["/constructor0/state_estimation"]):
        self.source_mcap = source_mcap
        self.topics = topics
        self.lap_data = get_lap_data(source_mcap, topics)

    def get_lap_data(self) -> list[CarState]:
        lst = []
        for msg_view in read_ros2_messages(
                DIR / self.source_mcap,
                self.topics,
        ):
            state = CarState.from_state_estimation(msg_view.ros_msg)
            lst.append(state)
        return lst


def _euclidean_dist(a: CarState, b: CarState) -> float:
    """2D distance between two CarState points."""
    return math.hypot(a.x - b.x, a.y - b.y)


def match_laps(
    source: list[CarState],
    target: list[CarState],
) -> dict[int, int]:
    """For each point in `source`, find the index of the closest point in `target`.

    Returns a dict mapping source_index -> target_index.
    """
    matched = {}
    for i, src_point in enumerate(source):
        closest_idx = min(
            range(len(target)),
            key=lambda j: _euclidean_dist(src_point, target[j]),
        )
        matched[i] = closest_idx
    return matched


def match_lap_states(
    source: list[CarState],
    target: list[CarState],
) -> dict[int, CarState]:
    """For each point in `source`, return the closest CarState from `target`.

    Returns a dict mapping source_index -> matched CarState.
    """
    index_map = match_laps(source, target)
    return {i: target[j] for i, j in index_map.items()}




if __name__ == "__main__":
    best = LapDataParser()
    ours = LapDataParser(source_mcap="data\hackathon\hackathon_good_lap.mcap")

    best_states = best.get_lap_data()
    ours_states = ours.get_lap_data()

    # Map each point in our lap to the closest point in the best lap
    matched = match_lap_states(ours_states, best_states)

    # Example: compare throttle at matched positions
    for i, (our_state, best_state) in enumerate(
        zip(ours_states, [matched[i] for i in range(len(ours_states))])
    ):
        delta_gas = best_state.gas - our_state.gas
        if abs(delta_gas) > 0.1:
            print(f"Point {i}: our gas={our_state.gas:.2f}, best gas={best_state.gas:.2f}, delta={delta_gas:+.2f}")
