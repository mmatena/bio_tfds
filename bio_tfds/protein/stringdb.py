import tensorflow as tf
import tensorflow_datasets as tfds

from bio_tfds.constants import DEFAULT_TFDS_DATA_DIR

_CITATION = R"""\
@article{von2005string,
  title={STRING: known and predicted protein--protein associations, integrated and transferred across organisms},
  author={Von Mering, Christian and Jensen, Lars J and Snel, Berend and Hooper, Sean D and Krupp, Markus and Foglierini, Mathilde and Jouffre, Nelly and Huynen, Martijn A and Bork, Peer},
  journal={Nucleic acids research},
  volume={33},
  number={suppl\_1},
  pages={D433--D437},
  year={2005},
  publisher={Oxford University Press}
}"""

_UNIPROT_ALIAS_KEY = "Ensembl_UniProt"

_ALIASES_DOWNLOAD = "https://stringdb-static.org/download/protein.aliases.v11.0.txt.gz"


class StringLinks(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    _DOWNLOAD_DIR = "https://stringdb-static.org/download/protein.links.v11.0.txt.gz"

    def __init__(self, data_dir=DEFAULT_TFDS_DATA_DIR, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Protein interactions.",
            features=tfds.features.FeaturesDict(
                {
                    # The accession number of the Uniprot entry for one protein.
                    "uniprot_acc_1": tfds.features.Text(),
                    # The accession number of the Uniprot entry for the other protein.
                    "uniprot_acc_2": tfds.features.Text(),
                    # Score of interaction. Higher values mean that there is more
                    # confidence in the prediction. In the range [0, 1].
                    "score": tf.float32,
                }
            ),
            homepage="https://string-db.org/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        files = dl_manager.download_and_extract(
            {
                "alias_file": _ALIASES_DOWNLOAD,
                "links_file": StringLinks._DOWNLOAD_URL,
            }
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=files,
            ),
        ]

    def _create_alias_map(self, alias_file):
        string_to_uniprot = {}
        with open(alias_file) as f:
            next(f)
            for line in f:
                items = line.split("\t")
                source = items[-1]
                if _UNIPROT_ALIAS_KEY in source.split():
                    string_name = items[0]
                    uniprot_ac = items[1]
                    string_to_uniprot[string_name] = uniprot_ac
        return string_to_uniprot

    def _generate_examples(self, alias_file, links_file):
        string_to_uniprot = self._create_alias_map(alias_file)

        with open(links_file) as f:
            next(f)
            for line in f:
                items = line.split(" ")
                p1, p2 = items[0], items[1]

                u1 = string_to_uniprot.get(p1, None)
                if not u1:
                    continue

                u2 = string_to_uniprot.get(p2, None)
                if not u2:
                    continue

                yield {
                    "uniprot_acc_1": u1,
                    "uniprot_acc_2": u2,
                    "score": float(items[-1]) / 1000.0,
                }
