from brake_analysis import BrakeBoundaryIssue, BrakeLevelIssue, detect_brake_plateaus, \
    analyze_brake_boundaries, analyze_brake_levels_in_mutual_plateaus, print_brake_recommendations, plot_brake_analysis, \
    _brake_build_boundary_events
from gas_analysis import ThrottleBoundaryIssue, ThrottleLevelIssue, detect_throttle_plateaus, \
    analyze_throttle_boundaries, analyze_throttle_levels_in_mutual_plateaus, print_gas_recommendations, \
    plot_gas_analysis, _gas_build_boundary_events
from parser import LapDataParser, align_laps, filter_arc_jumps
from steering_analysis import SteeringRecommendation, detect_steering_offsets, \
    group_steering_offsets, print_steering_recommendations, plot_steering_analysis
import os
import sys


def get_all_recommendations(
    file_fast: str,
    file_good: str,
    show: bool = False,
    save_dir: str | None = None,
) -> tuple[list, dict[str, str]]:
    """
    Returns (recommendations, plot_paths) where plot_paths is a dict like
    {"steering": "/tmp/.../steering.png", "throttle": "...", "brake": "..."}.
    If save_dir is provided, plots are always saved there.
    """

    # load
    if show:
        print("Loading reference (fast) lap …")
    ref_states = LapDataParser(file_fast).get_lap_data()
    if show:
        print("Loading slow lap …")
    slow_states = LapDataParser(file_good).get_lap_data()
    ref_states, slow_states = align_laps(ref_states, slow_states)

    # filter
    ref_count = len(ref_states)
    slow_count = len(slow_states)
    ref_states = filter_arc_jumps(ref_states)
    slow_states = filter_arc_jumps(slow_states)
    if show:
        print(f"{ref_count - len(ref_states)} ref points discarded")
        print(f"{slow_count - len(slow_states)} slot points discarded")

    # Determine save paths
    cli_save = sys.argv[1] if len(sys.argv) > 1 else None
    plot_paths: dict[str, str] = {}

    def _plot_path(name: str) -> str | None:
        if save_dir:
            p = os.path.join(save_dir, f"{name}.png")
            plot_paths[name] = p
            return p
        return cli_save

    # steering
    offsets, slow_to_ref = detect_steering_offsets(
        ref_states,
        slow_states,
    )
    steering_recommendations = group_steering_offsets(offsets)
    if show:
        print_steering_recommendations(steering_recommendations)

    plot_steering_analysis(
        ref_states,
        slow_states,
        steering_recommendations,
        slow_to_ref,
        save_path=_plot_path("steering"),
    )

    # gas
    ref_plateaus = detect_throttle_plateaus(ref_states)
    slow_plateaus = detect_throttle_plateaus(slow_states)
    events = _gas_build_boundary_events(ref_plateaus, slow_plateaus)
    gas_boundary_issues = analyze_throttle_boundaries(
        ref_plateaus,
        slow_plateaus,
        events,
    )
    gas_level_issues = analyze_throttle_levels_in_mutual_plateaus(
        ref_states,
        slow_states,
        events,
    )
    if show:
        print_gas_recommendations(gas_boundary_issues, gas_level_issues)

    plot_gas_analysis(
        ref_states,
        slow_states,
        ref_plateaus,
        slow_plateaus,
        gas_boundary_issues,
        gas_level_issues,
        save_path=_plot_path("throttle"),
    )

    # brake
    ref_plateaus = detect_brake_plateaus(ref_states)
    slow_plateaus = detect_brake_plateaus(slow_states)
    events = _brake_build_boundary_events(ref_plateaus, slow_plateaus)
    brake_boundary_issues = analyze_brake_boundaries(
        ref_plateaus,
        slow_plateaus,
        events,
    )
    brake_level_issues = analyze_brake_levels_in_mutual_plateaus(
        ref_states,
        slow_states,
        events,
    )
    if show:
        print_brake_recommendations(brake_boundary_issues, brake_level_issues)

    plot_brake_analysis(
        ref_states,
        slow_states,
        ref_plateaus,
        slow_plateaus,
        brake_boundary_issues,
        brake_level_issues,
        save_path=_plot_path("brake"),
    )

    all_recs = []
    all_recs.extend(steering_recommendations)
    all_recs.extend(gas_boundary_issues)
    all_recs.extend(gas_level_issues)
    all_recs.extend(brake_boundary_issues)
    all_recs.extend(brake_level_issues)
    return all_recs, plot_paths


if __name__ == "__main__":
    recs, paths = get_all_recommendations("data/hackathon/hackathon_fast_laps.mcap", "data/hackathon/hackathon_good_lap.mcap", show=True)
    print(f"\nTotal recommendations: {len(recs)}")
    print(f"Plot paths: {paths}")