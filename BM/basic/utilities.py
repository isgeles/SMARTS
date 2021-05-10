import math
from smarts.core.utils.math import squared_dist
from smarts.env.custom_observations import scan_for_vehicles
from typing import List

from smarts.core.sensors import Observation, VehicleObservation


def wait_on_front_vehicles(obs: Observation, chase_dist, filter_lane=None):
    vehicles: List[VehicleObservation] = scan_for_vehicles(
        None,
        -math.radians(10),
        math.radians(10),
        chase_dist + 10,
        obs.ego_vehicle_state,
        obs.neighborhood_vehicle_states,
    )

    if len(vehicles) > 0:
        if filter_lane:
            vehicles = filter(lambda v: v.lane_id == filter_lane, vehicles)

        vehicles = sorted(
            vehicles,
            key=lambda v: squared_dist(obs.ego_vehicle_state.position, v.position),
        )
        if len(vehicles) > 0:
            sq_dist = squared_dist(vehicles[0], obs.ego_vehicle_state.position)
            dist = math.sqrt(sq_dist)
            speed = obs.waypoint_paths[0][0].speed_limit * (chase_dist - dist - 2)
            return True, (max(speed, 0), 0)
    return False, None
