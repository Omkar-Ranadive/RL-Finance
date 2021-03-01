import pickle
from constants import DATA_PATH


def save_file(file, filename, path=DATA_PATH):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
        path (Path obj): Path to save file to
    """
    with open(path / filename, 'wb') as f:
        pickle.dump(file, f)


def load_file(filename, path=DATA_PATH):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
        path (Path obj): Path to load file from
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(path / filename, 'rb') as f:
        file = pickle.load(f)

    return file