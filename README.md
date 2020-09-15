# Biological TFDS Datasets
Some datasets that are useful for machine learning in biology.
We use [TensorFlow Datasets](https://www.tensorflow.org/datasets) to download and access the datasets.

## Usage

### Downloading a dataset
Just run the `download_and_prepare` method on the dataset builder.
Note that some of the datasets are very big and this can take some time.

Here is an example for the `UniRef50` dataset:
```python
from bio_tfds.protein import uniref

ds = uniref.UniRef50(data_dir="<data_dir>")
ds.download_and_prepare(download_dir="<download_dir>")
```

If you are on longleaf, the datasets have already been downloaded and prepared in the `/proj/craffel/datasets/tfds` directory.

### Accessing a dataset
In order to access a dataset that has already been prepared, use its [`as_dataset`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder#as_dataset) method.

Here is an example for the `UniRef50` dataset on longleaf:
```python
import itertools
from bio_tfds.protein import uniref

ds = uniref.UniRef50().as_dataset(split="train")

for x in itertools.islice(ds, 10):
    print(x)
```

This is very fast.
