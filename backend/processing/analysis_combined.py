from backend.processing import gas_analysis, brake_analysis
from backend.processing.brake_analysis import BrakeBoundaryIssue, BrakeLevelIssue, detect_brake_plateaus, \
    analyze_brake_boundaries, analyze_brake_levels_in_mutual_plateaus, print_brake_recommendations, plot_brake_analysis
from backend.processing.gas_analysis import ThrottleBoundaryIssue, ThrottleLevelIssue, detect_throttle_plateaus, \
    analyze_throttle_boundaries, analyze_throttle_levels_in_mutual_plateaus, print_gas_recommendations, \
    plot_gas_analysis
from backend.processing.parser import LapDataParser, align_laps, filter_arc_jumps
from backend.processing.steering_analysis import SteeringRecommendation, detect_steering_offsets, \
    group_steering_offsets, print_steering_recommendations, plot_steering_analysis
import sys


def get_all_recommendations(show: bool) -> list[ThrottleBoundaryIssue | ThrottleLevelIssue | BrakeBoundaryIssue | BrakeLevelIssue | SteeringRecommendation]:

    # load
    print("Loading reference (fast) lap …")
    ref_states = LapDataParser("data/hackathon/hackathon_fast_laps.mcap").get_lap_data()
    print("Loading slow lap …")
    slow_states = LapDataParser("data/hackathon/hackathon_good_lap.mcap").get_lap_data()
    ref_states, slow_states = align_laps(ref_states, slow_states)


    # filter
    ref_count = len(ref_states)
    slow_count = len(slow_states)
    ref_states = filter_arc_jumps(ref_states)
    slow_states = filter_arc_jumps(slow_states)
    print(f"{ref_count - len(ref_states)} ref points discarded")
    print(f"{slow_count - len(slow_states)} slot points discarded")

    save = sys.argv[1] if len(sys.argv) > 1 else None


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
            save_path=save,
        )


    # gas
    ref_plateaus = detect_throttle_plateaus(ref_states)
    slow_plateaus = detect_throttle_plateaus(slow_states)
    events = gas_analysis._build_boundary_events(ref_plateaus, slow_plateaus)
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
        )


    # brake
    ref_plateaus = detect_brake_plateaus(ref_states)
    slow_plateaus = detect_brake_plateaus(slow_states)
    events = brake_analysis._build_boundary_events(ref_plateaus, slow_plateaus)
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
        )

    return []


if __name__ == "__main__":
    get_all_recommendations(True)