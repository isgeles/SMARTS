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

try:
    copy_to_dir(scenario_map_file, s_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e

ego_missions = [
    t.Mission(
        route=t.Route(
            begin=("straightaway", 1, 1),
            end=("straightaway", 0, "max"),
        ),
        via=[
            t.Via("straightaway", 1, 70, 12.9),
            t.Via("straightaway", 0, 80, 12.9),
            t.Via("straightaway", 0, 160, 12.9),
            t.Via("straightaway", 1, 180, 14),
            t.Via("straightaway", 1, 360, 16),
            t.Via("straightaway", 0, 480, 16),
        ],
    )
]

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("straightaway", 0, 0),
                end=("straightaway", 0, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("ego", speed=t.Distribution(mean=1, sigma=0)): 1},
        ),
        t.Flow(
            route=t.Route(
                begin=("straightaway", 1, 30),
                end=("straightaway", 1, "max"),
            ),
            rate=1,
            actors={
                t.TrafficActor(
                    "left_lane_hog",
                    speed=t.Distribution(mean=0.75, sigma=0),
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
