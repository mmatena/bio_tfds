"""Protein datasets that come from joining other datasets here."""

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from bio_tfds.constants import DEFAULT_TFDS_DATA_DIR


class UniRef50WithPfamRegions(tfds.core.GeneratorBasedBuilder):
    """UnirRef50 sequences with Pfam regions annotated."""

    pass
