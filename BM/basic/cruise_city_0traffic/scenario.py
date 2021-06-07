import logging
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

from sys import path

file_path = Path(__file__)
path.append(str(file_path.parent.parent))
from copy_scenario import copy_to_dir

scenario_map_file = "maps/city/toronto_downtown"

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
        t.Route(
            begin=("10334991", 0, 20), 
            end=("-263506114#7", 0, "max"),
            via=("8101761",),
        ),
    )
]

scenario = t.Scenario(
    ego_missions=ego_missions,
)

try:
    copy_to_dir(scenario_map_file, output_dir)
except Exception as e:
    logger.error(f"Scenario {scenario_map_file} failed to copy")
    raise e
gen_scenario(scenario, output_dir=output_dir, overwrite=True)
