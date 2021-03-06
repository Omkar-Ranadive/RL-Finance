from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
META_PATH = PARENT_PATH / '../meta'
MODEL_PATH = PARENT_PATH / 'models'

HIST_PATH = DATA_PATH  # These files are huge, so change the path according

# Agent constants
MAX_EPSILON = 1.0
MIN_EPSILON = 0.1