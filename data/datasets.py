import clrs
import os
import shutil
import time
from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
import requests
import tensorflow as tf


def download_dataset():
  """Downloads CLRS30 dataset if not already downloaded."""
  dataset_folder = os.path.join('/tmp/CLRS30', clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)
  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join('/tmp/CLRS30', os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder
