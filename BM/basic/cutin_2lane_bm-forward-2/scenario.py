import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

file_path = Path(__file__)
path.append(str(file_path.parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "maps/straight/2lane_ow_straight"

logger = logging.getLogger(str(file_path))

scenario_name= str(file_path.parent.name)
s_dir = str(file_path.parent)
output_dir = f"{str(file_path.parent.parent)}/scenarios/{scenario_name}"


base_offset = 20
ego_missions = [
    t.Mission(
        route=t.Route(
            begin=("straightaway", 1, 1 + base_offset),
            end=("straightaway", 0, "max"),
        ),
        via=[
            t.Via("straightaway", 1, 60, 20),
            t.Via("straightaway", 0, 100, 15),
            t.Via("straightaway", 0, 120, 7.5),
        ],
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("straightaway", 0, 0 + base_offset),
                end=("straightaway", 0, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("ego", speed=t.Distribution(mean=1, sigma=0)): 1},
        ),
        t.Flow(
            route=t.Route(
                begin=("straightaway", 1, 50 + base_offset),
                end=("straightaway", 1, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "left_lane_hog",
                    speed=t.Distribution(mean=0.90, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        ),
        t.Flow(
            route=t.Route(
                begin=("straightaway", 0, 40 + base_offset),
                end=("straightaway", 0, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "forward_car",
                    speed=t.Distribution(mean=1, sigma=0),
                    lane_changing_model=t.LaneChangingModel(
                        strategic=0, cooperative=0, keepRight=0
                    ),
                ): 1
            },
        ),
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
