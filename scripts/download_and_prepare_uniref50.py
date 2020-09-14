from bio_tfds.protein import uniref

ds = uniref.UniRef50(data_dir="/pine/scr/m/m/mmatena/tfds_data")
ds.download_and_prepare(download_dir="/pine/scr/m/m/mmatena/tfds_data/downloads")
