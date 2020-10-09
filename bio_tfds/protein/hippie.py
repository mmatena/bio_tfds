"""HIPPIE v2 protein-protein interaction dataset."""
import csv
import io
import ssl
import time
import urllib.parse
import urllib.request

from Bio import SeqIO

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from bio_tfds.constants import DEFAULT_TFDS_DATA_DIR

_DOWNLOAD_URL = (
    "http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/HIPPIE-current.mitab.txt"
)

_DESCRIPTION = R"""\
Protein-protein interaction dataset."""

_CITATION = R"""\
@article{alanis2016hippie,
  title={HIPPIE v2. 0: enhancing meaningfulness and reliability of protein--protein interaction networks},
  author={Alanis-Lobato, Gregorio and Andrade-Navarro, Miguel A and Schaefer, Martin H},
  journal={Nucleic acids research},
  pages={gkw985},
  year={2016},
  publisher={Oxford University Press}
}"""


def _parse_accession(s):
    splits = s.split(":")
    if len(splits) != 2 or splits[0] != "uniprotkb":
        return None
    return splits[1]


def _extract_example(example):
    return {
        "protein_a_identifier": _parse_accession(example["Alt IDs Interactor A"]),
        "protein_b_identifier": _parse_accession(example["Alt IDs Interactor B"]),
        "confidence": float(example["Confidence Value"]),
    }


def _send_seq_request(accs, max_tries=10):
    url = "https://www.uniprot.org/uploadlists/"
    payload = {
        "from": "ACC+ID",
        "to": "ACC",
        "format": "fasta",
        "query": " ".join(accs),
    }

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    data = urllib.parse.urlencode(payload).encode("utf-8")

    for attempt in range(max_tries):
        try:
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req, context=ctx) as f:
                return f.read().decode("utf-8")
        except urllib.error.URLError:
            print(f"{2**attempt} s backoff")
            time.sleep(2 ** attempt)
    raise urllib.error.URLError


def _retrieve_sequences_from_remote(accs):
    ret = {}
    if not accs:
        return ret

    response = _send_seq_request(accs)
    for seq in SeqIO.parse(io.StringIO(response), "fasta"):
        acc = seq.name.split("|")[-1]
        ret[acc] = str(seq.seq)

    missing_accs = set(acc for acc in accs if acc not in ret)
    return ret, missing_accs


_VERSION = tfds.core.Version("1.0.0")


class MhcBindingAffinityConfig(tfds.core.BuilderConfig):
    def __init__(self, *, include_sequences=False, **kwargs):
        super().__init__(description=_DESCRIPTION, version=_VERSION, **kwargs)
        self.include_sequences = include_sequences


class Hippie(tfds.core.GeneratorBasedBuilder):
    """The HIPPIE dataset."""

    VERSION = tfds.core.Version("1.0.0")

    UNSTABLE = "Looks like the we can't get a fixed link to the version that is current at download time."

    BUILDER_CONFIGS = [
        MhcBindingAffinityConfig(name="no_seq", include_sequences=False),
        MhcBindingAffinityConfig(name="with_seq", include_sequences=True),
    ]

    def __init__(self, data_dir=DEFAULT_TFDS_DATA_DIR, **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self):
        features = {
            # The UniprotKb accessions of the proteins involved in the interaction.
            "protein_a_identifier": tfds.features.Text(),
            "protein_b_identifier": tfds.features.Text(),
            # Confidence score of the interaction. I think this is between 0 and 1.
            "confidence": tf.float32,
        }
        if self.builder_config.include_sequences:
            features.update(
                {
                    # The amino acid sequences of the proteins involved in the interaction.
                    "protein_a_sequence": tfds.features.Text(),
                    "protein_b_sequence": tfds.features.Text(),
                }
            )
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict(features),
            homepage="http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/index.php",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "mitab_file": extracted_path,
                },
            ),
        ]

    def _generate_examples(self, mitab_file):
        if self.builder_config.include_sequences:
            yield from self._generate_examples_with_seq(mitab_file)
        else:
            yield from self._generate_examples_no_seq(mitab_file)

    def _generate_examples_no_seq(self, mitab_file):
        with tf.io.gfile.GFile(mitab_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for index, row in enumerate(reader):
                example = _extract_example(row)
                yield index, example

    def _generate_examples_with_seq(self, mitab_file):
        max_to_fetch = 40000
        seq_cache = {}
        example_queue = []
        accs_to_fetch = set()
        missing_accs = set()

        def add_seqs_to_example(example):
            a_acc, b_acc = (
                example["protein_a_identifier"],
                example["protein_b_identifier"],
            )
            example["protein_a_sequence"] = seq_cache.get(a_acc, None)
            example["protein_b_sequence"] = seq_cache.get(b_acc, None)
            return example

        def flush_caches():
            nonlocal example_queue, accs_to_fetch
            print("Retrieving sequences")
            seqs, missing = _retrieve_sequences_from_remote(accs_to_fetch)
            seq_cache.update(seqs)
            missing_accs.update(missing)
            print("Done retrieving sequences")
            for index, example in example_queue:
                example = add_seqs_to_example(example)
                if (
                    not example["protein_a_sequence"]
                    or not example["protein_b_sequence"]
                ):
                    continue
                yield index, example
            example_queue = []
            accs_to_fetch = set()

        def process_example(index, example):
            a_acc, b_acc = (
                example["protein_a_identifier"],
                example["protein_b_identifier"],
            )

            if a_acc not in seq_cache:
                if a_acc not in missing_accs:
                    accs_to_fetch.add(a_acc)
            if b_acc not in seq_cache:
                if b_acc not in missing_accs:
                    accs_to_fetch.add(b_acc)

            if a_acc in seq_cache and b_acc in seq_cache:
                yield index, add_seqs_to_example(example)
            else:
                example_queue.append((index, example))

            if len(accs_to_fetch) >= max_to_fetch:
                yield from flush_caches()

        with tf.io.gfile.GFile(mitab_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for index, row in enumerate(reader):
                example = _extract_example(row)
                if (
                    not example["protein_a_identifier"]
                    or not example["protein_b_identifier"]
                ):
                    continue
                yield from process_example(index, example)

        yield from flush_caches()
