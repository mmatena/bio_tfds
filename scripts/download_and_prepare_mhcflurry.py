from bio_tfds.constants import DEFAULT_TFDS_DOWNLOAD_DIR
from bio_tfds.mhc import mhcflurry

import tensorflow_datasets as tfds

checksum_dir = "/nas/longleaf/home/mmatena/projects/bio_tfds/url_checksums"
# checksum_dir = "./url_checksums"
tfds.download.add_checksums_dir(checksum_dir)

ds = mhcflurry.MhcBindingAffinity(config="aligned_mhc")
ds.download_and_prepare(
    download_dir=DEFAULT_TFDS_DOWNLOAD_DIR,
    download_config=tfds.download.DownloadConfig(register_checksums=True),
)
