import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

file_path = Path(__file__)
path.append(str(file_path.parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "maps/straight/6lane_straight"

logger = logging.getLogger(str(file_path))

scenario_name= str(file_path.parent.name)
s_dir = str(file_path.parent)
output_dir = f"{str(file_path.parent.parent)}/scenarios/{scenario_name}"

ego_missions = [
    t.EndlessMission(
        begin=("-straightaway", 2, 200),
    )
]

start_offset = 10
start_offset_2 = 40
interval = 10
traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("straightaway", 2, o * interval + start_offset_2),
                end=("straightaway", 2, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "car",
                    speed=t.Distribution(mean=0.9, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        )
        for o in range(4)
    ]
    + [
        t.Flow(
            route=t.Route(
                begin=("straightaway", 2, start_offset),
                end=("straightaway", 2, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "ego",
                    speed=t.Distribution(mean=0.9, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
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
