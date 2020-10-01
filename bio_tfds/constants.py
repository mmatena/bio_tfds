"""Constants used throughout the codebase."""
import os


def _is_longleaf():
    """See if we are on the longleaf cluster."""
    return os.path.exists("/nas") and os.path.exists("/pine")


if _is_longleaf():
    # The default directory to use as data_dir for tfds.
    DEFAULT_TFDS_DATA_DIR = "/proj/craffel/datasets/tfds"

    # The default directory to download files to for tfds.
    DEFAULT_TFDS_DOWNLOAD_DIR = "/proj/craffel/datasets/tfds/downloads"
else:
    DEFAULT_TFDS_DATA_DIR = None
    DEFAULT_TFDS_DOWNLOAD_DIR = None
