"""Lap data parsing and processing logic."""


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
