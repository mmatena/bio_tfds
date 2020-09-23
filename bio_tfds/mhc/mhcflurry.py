"""TFDS versions of some of the data used to train MHCflurry 2.0.1.

Taken from https://github.com/iskandr/cd8-tcell-epitope-prediction-data
"""
import csv
from enum import Enum

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from bio_tfds.constants import DEFAULT_TFDS_DATA_DIR

_CITATION = R"""\
@article{o2020mhcflurry,
  title={MHCflurry 2.0: Improved Pan-Allele Prediction of MHC Class I-Presented Peptides by Incorporating Antigen Processing},
  author={O’Donnell, Timothy J and Rubinsteyn, Alex and Laserson, Uri},
  journal={Cell Systems},
  volume={11},
  number={1},
  pages={42--48},
  year={2020},
  publisher={Elsevier}
}
@article{o2018mhcflurry,
  title={MHCflurry: open-source class I MHC binding affinity prediction},
  author={O'Donnell, Timothy J and Rubinsteyn, Alex and Bonsack, Maria and Riemer, Angelika B and Laserson, Uri and Hammerbacher, Jeff},
  journal={Cell systems},
  volume={7},
  number={1},
  pages={129--132},
  year={2018},
  publisher={Elsevier}
}"""

_MHC_BINDING_AFFINITY_DESC = R"""\
Data for building predictive models of CD8+ T-cell epitopes,
including peptide-MHC binding affinity."""

_MEASUREMENT_INEQUALITIES = ("=", "<", ">")


class MhcflurrySpecies(Enum):
    """All of the species in the dataset.

    The enum name is a colloquial name for the species
    while its value is the prefix used in the dataset.
    """

    @classmethod
    def parse(cls, x):
        if isinstance(x, cls):
            return x
        else:
            return getattr(cls, x.upper())

    # These counts only consider equality data.
    HUMAN = "HLA"  # 100304 examples
    #                   - HLA-A: 72570
    #                   - HLA-B: 26478
    #                   - HLA-C: 1219
    CHIMPANZEE = "Patr"  # 2995 examples
    GORILLA = "Gogo"  # 15 examples
    RHESUS_MACAQUE = "Mamu"  # 10649 examples
    HORSE = "Eqca"
    COW = "BoLA"
    MOUSE = "H-2"
    PIG = "SLA"


_PEP_MHC_AFFINITY_URL = "https://raw.githubusercontent.com/iskandr/cd8-tcell-epitope-prediction-data/master/mhcflurry-training-data/peptide-mhc-binding-affinity.csv"
_MHC_SEQUENCE_URL = "https://raw.githubusercontent.com/iskandr/cd8-tcell-epitope-prediction-data/master/mhc-sequences/class1_mhc_sequences.csv"


def normalize_ic50(ic50):
    return 1.0 - tf.math.log(tf.minimum(tf.maximum(ic50, 1.0), 50000.0)) / tf.math.log(
        50000.0
    )


def species_from_allele(allele):
    species = tf.strings.split(allele, sep="-")
    species = species[..., :-1]
    species = tf.strings.reduce_join(species, axis=-1, separator="-")
    return species


def gene_from_allele(allele):
    species = tf.strings.split(allele, sep="*")
    species = species[..., 0]
    return species


class MhcBindingAffinity(tfds.core.GeneratorBasedBuilder):
    """Dataset about the binding affinity of peptides to MHC sequences.

    Note that 2699 out of 126645 data points are not included as we do
    not have the sequences for their alleles.
    """

    VERSION = tfds.core.Version("1.0.0")

    def __init__(
        self,
        normalize_measurement=True,
        include_inequalities=False,
        species=None,
        genes=None,
        exclude_genes=None,
        data_dir=DEFAULT_TFDS_DATA_DIR,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.include_inequalities = include_inequalities
        self.normalize_measurement = normalize_measurement
        if species and not isinstance(species, (list, tuple)):
            species = [species]
        self.species = (
            species if not species else [MhcflurrySpecies.parse(s) for s in species]
        )
        if genes and not isinstance(genes, (list, tuple)):
            genes = [genes]
        self.genes = genes
        if exclude_genes and not isinstance(exclude_genes, (list, tuple)):
            exclude_genes = [exclude_genes]
        self.exclude_genes = exclude_genes

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_MHC_BINDING_AFFINITY_DESC,
            features=tfds.features.FeaturesDict(
                {
                    # Name of MHC allele, e.g. "HLA-A*02:01".
                    "mhc_allele": tfds.features.Text(),
                    # nM affinity (smaller is better), most often an
                    # IC50 (inhibitory concentration). If `normalize_measurement` is True
                    # then this will be converted to a value between 0 and 1 through the
                    # transformation 1 - log(min(max(affinity, 1), 50000))/log(50000).
                    "affinity": tf.float32,
                    # One of {=, >, <}. Most often = but < indicates that the measurement
                    # is an upper bound (and a lower bound >). If `include_inequalities`
                    # is False, then only equalities will be included.
                    "measurement_inequality": tfds.features.ClassLabel(
                        names=_MEASUREMENT_INEQUALITIES
                    ),
                    # Amino acid sequence of the peptide.
                    "peptide_sequence": tfds.features.Text(),
                    # Most of the diversity of Class I MHCs occurs in exons 2 and 3,
                    # so some sequences are limited to those regions.
                    "mhc_sequence": tfds.features.Text(),
                }
            ),
            homepage="https://github.com/iskandr/cd8-tcell-epitope-prediction-data",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        extracted_paths = dl_manager.download(
            {
                "affinity_file": _PEP_MHC_AFFINITY_URL,
                "mhc_sequence_file": _MHC_SEQUENCE_URL,
            }
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=extracted_paths,
            ),
        ]

    def _create_allele_to_sequence_map(self, mhc_sequence_file):
        allele_to_sequence = {}
        with open(mhc_sequence_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                allele_to_sequence[row["name"]] = row["seq"]
        return allele_to_sequence

    def _generate_examples(self, affinity_file, mhc_sequence_file):
        allele_to_sequence = self._create_allele_to_sequence_map(mhc_sequence_file)

        missing_seqs = 0

        with open(affinity_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for index, row in enumerate(reader):
                allele = row["allele"]
                mhc_sequence = allele_to_sequence.get(allele, None)

                if mhc_sequence is None:
                    missing_seqs += 1
                    continue

                yield index, {
                    "mhc_allele": allele,
                    "affinity": float(row["measurement_value"]),
                    "measurement_inequality": row["measurement_inequality"],
                    "peptide_sequence": row["peptide"],
                    "mhc_sequence": mhc_sequence,
                }
        if missing_seqs:
            print(
                f"We were unable to find sequences for {missing_seqs} affinity data points."
            )

    def _filter_inequalities_fn(self, x):
        return tf.equal(
            x["measurement_inequality"], _MEASUREMENT_INEQUALITIES.index("=")
        )

    def _normalize_measurement_fn(self, x):
        x["affinity"] = normalize_ic50(x["affinity"])
        return x

    def _filter_species_fn(self, x):
        x_species = species_from_allele(x["mhc_allele"])
        return tf.reduce_any([tf.equal(s.value, x_species) for s in self.species])

    def _filter_genes_fn(self, x):
        x_gene = gene_from_allele(x["mhc_allele"])
        return tf.reduce_any([tf.equal(s, x_gene) for s in self.genes])

    def _filter_exclude_genes_fn(self, x):
        x_gene = gene_from_allele(x["mhc_allele"])
        return tf.reduce_all([tf.not_equal(s, x_gene) for s in self.exclude_genes])

    def _as_dataset(self, *args, **kwargs):
        ds = super()._as_dataset(*args, **kwargs)
        if not self.include_inequalities:
            ds = ds.filter(self._filter_inequalities_fn)
        if self.normalize_measurement:
            ds = ds.map(
                self._normalize_measurement_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        if self.species is not None:
            ds = ds.filter(self._filter_species_fn)
        if self.genes is not None:
            ds = ds.filter(self._filter_genes_fn)
        if self.exclude_genes is not None:
            ds = ds.filter(self._filter_exclude_genes_fn)
        return ds
