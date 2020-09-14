from bio_tfds.constants import DEFAULT_TFDS_DOWNLOAD_DIR
from bio_tfds.protein import pfam

ds = pfam.PfamARegionsUniprot()
ds.download_and_prepare(download_dir=DEFAULT_TFDS_DOWNLOAD_DIR)
