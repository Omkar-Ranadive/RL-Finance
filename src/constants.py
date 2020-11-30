from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
META_PATH = PARENT_PATH / '../meta'


# HIST files are huge so data is not kept on a relative path. Change it according to your system
HIST_PATH = Path(r'W:/Finance_HIST/Feeds')

# Agent constants
MAX_EPSILON = 1.0
MIN_EPSILON = 0.1