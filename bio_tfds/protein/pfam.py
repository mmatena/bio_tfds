"""Datasets from Pfam."""
import csv

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from bio_tfds.constants import DEFAULT_TFDS_DATA_DIR


_CITATION = R"""\
@article{bateman2004pfam,
  title={The Pfam protein families database},
  author={Bateman, Alex and Coin, Lachlan and Durbin, Richard and Finn, Robert D and Hollich, Volker and Griffiths-Jones, Sam and Khanna, Ajay and Marshall, Mhairi and Moxon, Simon and Sonnhammer, Erik LL and others},
  journal={Nucleic acids research},
  volume={32},
  number={suppl\_1},
  pages={D138--D141},
  year={2004},
  publisher={Oxford University Press}
}
"""


class PfamARegionsUniprot(tfds.core.GeneratorBasedBuilder):
    """The PfamA Regions Uniprot dataset."""

    VERSION = tfds.core.Version("1.0.0")

    _DOWNLOAD_URL = "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam33.1/Pfam-A.regions.uniprot.tsv.gz"

    def __init__(self, data_dir=DEFAULT_TFDS_DATA_DIR, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="All PfamA regions with their respective Uniprot entries.",
            features=tfds.features.FeaturesDict(
                {
                    # The accession number of the Uniprot entry.
                    "uniprot_acc": tfds.features.Text(),
                    # The accession number of the PfamA entry.
                    "pfam_acc": tfds.features.Text(),
                    # The sequence version. Of what? I don't know. Probably
                    # the Uniprot.
                    "seq_version": tf.int32,
                    # The 0-BASED, INCLUSIVE start index of the region in the
                    # protein's AA sequence. Note that this is different than
                    # in the raw data, which is 1-based and inclusive.
                    "start": tf.int32,
                    # The 0-BASED, EXCLUSIVE end index of the region in the
                    # protein's AA sequence. The raw data is 1-based and
                    # inclusive, which works out to be the same.
                    "end": tf.int32,
                }
            ),
            homepage="https://pfam.xfam.org/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(
            PfamARegionsUniprot._DOWNLOAD_URL
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "tsv_file": extracted_path,
                },
            ),
        ]

    def _generate_examples(self, fasta_file):
        with tf.io.gfile.GFile(fasta_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for index, row in enumerate(reader):
                example = {
                    "uniprot_acc": row["uniprot_acc"],
                    "pfam_acc": row["pfamA_acc"],
                    "seq_version": int(row["seq_version"]),
                    "start": int(row["seq_start"]) - 1,
                    "end": int(row["seq_end"]),
                }
                yield index, example
