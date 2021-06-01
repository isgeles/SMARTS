import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

file_path = Path(__file__)
path.append(str(file_path.parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "maps/cross/4lane_cross"

logger = logging.getLogger(str(file_path))

scenario_name= str(file_path.parent.name)
s_dir = str(file_path.parent)
output_dir = f"{str(file_path.parent.parent)}/scenarios/{scenario_name}"

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    t.Mission(
        t.Route(begin=("east_ew", 0, 20), end=("west_ew", 1, "max")),
        via=[
            t.Via("east_ew", 0, 50, 30),
            t.Via(t.JunctionEdgeIDResolver("east_ew", 0, "west_ew", 1), 1, 5, 15),
            t.Via("west_ew", 1, 20, 7.5),
        ],
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("east_ew", 1, 20),
                end=("west_ew", 1, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor("ego", speed=t.Distribution(mean=0.9, sigma=0)): 1
            },
        )
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_missions,
)

try:
    copy_to_dir(scenario_map_file, output_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e
gen_scenario(scenario, output_dir=output_dir, overwrite=True)
