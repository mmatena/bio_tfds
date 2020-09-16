#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/bio_tfds
#############################################################


module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo "python $PROJECT_DIR/scripts/download_and_prepare_stringdb.py"
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=20 \
    --time=5- \
    --mem=48g \
    --partition=general \
    --wrap="\"$(run_python)\"")
  eval $CMD
}


# Run the command to actually launch the job.
launch

