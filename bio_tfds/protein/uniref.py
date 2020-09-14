"""The UniRef50 dataset.

To download the files, please go to
https://www.uniprot.org/uniref/?query=identity:0.5 and download all of the
sequences as both FASTA and tab-separated.
"""
from Bio import SeqIO

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DOWNLOAD_URL = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref50/uniref50.fasta.gz"

_DESCRIPTION = r"""\
The UniProt Reference Clusters (UniRef) provide clustered sets of sequences
from the UniProt Knowledgebase (including isoforms) and selected UniParc records.
This hides redundant sequences and obtains complete coverage of the sequence space
at three resolutions:
UniRef100 combines identical sequences and sub-fragments with 11 or more residues from
any organism into a single UniRef entry.
UniRef90 is built by clustering UniRef100 sequences such that each cluster is composed of sequences
that have at least 90% sequence identity to, and 80% overlap with, the longest sequence (a.k.a. seed sequence).
UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to,
and 80% overlap with, the longest sequence in the cluster."""

_CITATION = """\
@article{suzek2007uniref,
  title={UniRef: comprehensive and non-redundant UniProt reference clusters},
  author={Suzek, Baris E and Huang, Hongzhan and McGarvey, Peter and Mazumder, Raja and Wu, Cathy H},
  journal={Bioinformatics},
  volume={23},
  number={10},
  pages={1282--1288},
  year={2007},
  publisher={Oxford University Press}
}"""


def _get_cluster_name(desc_list):
    # The first element in the desc is the unique_identifier. It is followed by
    # the cluster name.
    cluster_name = []
    for word in desc_list[1:]:
        if word.startswith("n="):
            break
        cluster_name.append(word)
    return " ".join(cluster_name)


def _get_word_by_prefix(desc_list, prefix):
    for word in desc_list:
        if word.startswith(prefix):
            return word[len(prefix):]


def _get_tax_name(desc_list):
    active = False
    tax_name = []
    for word in desc_list:
        if word.startswith("Tax="):
            active = True
            tax_name.append(word[4:])
        elif active and "=" in word:
            break
        elif active:
            tax_name.append(word)
    return " ".join(tax_name)


def _extract_example(seq):
    # Here is an example of what the description will look like:
    # UniRef50_Q8WZ42 Titin n=1336 Tax=Vertebrata TaxID=7742 RepID=TITIN_HUMAN
    desc = seq.description.split()
    return {
        "unique_identifier": seq.name,
        "cluster_name": _get_cluster_name(desc),
        "num_members": int(_get_word_by_prefix(desc, "n=")),
        "tax_name": _get_tax_name(desc),
        "tax_id": _get_word_by_prefix(desc, "TaxID="),
        "representative_member": _get_word_by_prefix(desc, "RepID="),
        "aa_sequence": str(seq.seq),
    }


class UniRef50(tfds.core.GeneratorBasedBuilder):
    """The UniRef50 Dataset."""

    VERSION = tfds.core.Version('1.0.0')

    UNSTABLE = "The current_release is updated every 8 weeks."

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # The primary accession number of the UniRef cluster.
                "unique_identifier": tfds.features.Text(),
                # The name of the UniRef cluster.
                "cluster_name": tfds.features.Text(),
                # The number of UniRef cluster members.
                "num_members": tf.int32,
                # The scientific name of the lowest common taxon shared by all
                # UniRef cluster members.
                "tax_name": tfds.features.Text(),
                # Thee id of the lowest common taxon shared by all
                # UniRef cluster members.
                "tax_id": tfds.features.Text(),
                # The entry name of the representative member of the
                # UniRef cluster.
                "representative_member": tfds.features.Text(),
                # The uppercase AA sequence of the protein.
                'aa_sequence': tfds.features.Text(),
            }),
            homepage='https://www.uniprot.org/help/uniref',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        if True:
            raise ValueError("Point to the correct path.")
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "fasta_file": extracted_path,
                },
            ),
        ]

    def _generate_examples(self, fasta_file):
        with tf.io.gfile.GFile(fasta_file) as f:
            for seq in SeqIO.parse(f, "fasta"):
                example = _extract_example(seq)
                yield example["unique_identifier"], example
