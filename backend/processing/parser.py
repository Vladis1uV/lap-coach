"""Lap data parsing and processing logic."""

from mcap_ros2.reader import read_ros2_messages
import pathlib
import os

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
        os.path.join(DIR, source_mcap),
        topics,
    ):
        msg = msg_view.ros_msg
        print("x:", msg.x_m)
        print("y:", msg.y_m)
        print("yaw:", msg.yaw_rad)
        print("brake:", msg.brake)
        print("gas:", msg.gas)

if __name__ == "__main__":
    get_lap_data()
