import logging
import math

import gym
import numpy as np

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.core.utils.math import (
    min_angles_difference_signed,
)
from smarts.core.waypoints import Waypoint, Waypoints

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class CutinAgent(Agent):
    def __init__(self):
        self.lane_index = 1
        self._initial_heading = 0
        self._task_is_triggered = False
        self._counter = 0
        self.lateral_gain = 0.34
        self.heading_gain = 1.2
        self._des_speed = 12

    def act(self, obs: Observation):
        aggressiveness = 10

        vehicle = self.sim._vehicle_index.vehicles_by_actor_id("Agent-007")[0]

        miss = self.sim._vehicle_index.sensor_state_for_vehicle_id(
            vehicle.id
        ).mission_planner

        neighborhood_vehicles = obs.neighborhood_vehicle_states
        neighborhood_vehicles = [
            vehicle for vehicle in neighborhood_vehicles if "ego" in vehicle.id
        ]

        pose = vehicle.pose
        position = pose.position[:2]
        start_lane = miss._road_network.nearest_lane(
            miss._mission.start.position,
            include_junctions=False,
            include_special=False,
        )
        offset = miss._road_network.offset_into_lane(start_lane, pose.position[:2])

        if self._task_is_triggered is False:
            self.lane_index = start_lane.getID().split("_")[-1]

        if len(neighborhood_vehicles) < 1:
            fff = miss._waypoints.waypoint_paths_on_lane_at(
                position, start_lane.getID(), 60
            )[0]
            speed_difference = 0
            position_adjust = 0.2

        else:
            target_vehicle = neighborhood_vehicles[0]
            speed_difference = obs.ego_vehicle_state.speed - (target_vehicle.speed)
            target_p = target_vehicle.position[0:2]
            target_l = miss._road_network.nearest_lane(target_p)

            target_offset = miss._road_network.offset_into_lane(target_l, target_p)
            fq = offset - target_offset

            off_des = (aggressiveness / 10) * 15 + (1 - aggressiveness / 10) * 25

            if abs(fq - off_des) > 1 and self._task_is_triggered is False:
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, start_lane.getID(), 60
                )[0]
                position_adjust = -0.3 * (fq - off_des)
            elif self._counter < 5:
                self._task_is_triggered = True
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, start_lane.getID(), 60
                )[0]
                position_adjust = -0.3 * (fq - off_des)
                self._counter += 1
                self.lateral_gain = 0.1
                self.heading_gain = 2.1
            else:
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, target_l.getID(), 60
                )[0]
                lat_error = fff[0].signed_lateral_error(
                    [vehicle.position[0], vehicle.position[1]]
                )
                if abs(lat_error) < 0.3:
                    self.lateral_gain = 0.34
                    self.heading_gain = 1.2
                self._task_is_triggered = True
                position_adjust = -0.3 * (fq - off_des)

        look_ahead_wp_num = 3
        look_ahead_dist = 3
        vehicle_look_ahead_pt = [
            obs.ego_vehicle_state.position[0]
            - look_ahead_dist * math.sin(obs.ego_vehicle_state.heading),
            obs.ego_vehicle_state.position[1]
            + look_ahead_dist * math.cos(obs.ego_vehicle_state.heading),
        ]

        reference_heading = fff[look_ahead_wp_num].heading
        heading_error = min_angles_difference_signed(
            (obs.ego_vehicle_state.heading % (2 * math.pi)), reference_heading
        )
        controller_lat_error = fff[look_ahead_wp_num].signed_lateral_error(
            vehicle_look_ahead_pt
        )

        steer = (
            self.lateral_gain * controller_lat_error + self.heading_gain * heading_error
        )
        throttle = (
            -0.43 * (speed_difference)
            - 1.1 * abs(obs.ego_vehicle_state.linear_velocity[1])
            + position_adjust
        )

        if throttle >= 0:
            brake = 0
        else:
            brake = abs(throttle)
            throttle = 0

        return (throttle, brake, steer)


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering,
            max_episode_steps=max_episode_steps,
            neighborhood_vehicles=NeighborhoodVehicles(850),
        ),
        agent_builder=CutinAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )
    CutinAgent.sim = env._smarts

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        agent.sim = env._smarts
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
