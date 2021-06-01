import os
import shutil

from pathlib import Path
from smarts.sstudio import types

dir = str(Path.absolute(Path(__file__).parent))


def copy_to_dir(scenario_map_file, scenario_directory):
    success = False

    basename = os.path.basename(scenario_map_file)
    try:
        os.mkdir(scenario_directory)
    except FileExistsError:
        pass
    shutil.copyfile(
        f"{dir}/{scenario_map_file}.net.xml", f"{scenario_directory}/map.net.xml"
    )

    return success
