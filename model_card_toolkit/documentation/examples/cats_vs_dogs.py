# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cats vs Dogs utilities.

A collection of helper functions for the Cats vs Dogs model used in
Standalone_Model_Card_Toolkit_Demo.ipynb.
"""

from typing import Any, Dict, Text
import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = 150
BATCH_SIZE = 32
NUM_BATCHES = 10


def get_data() -> Dict[Text, Any]:
  """Return 320 examples from Cats vs Dogs dataset.

  Returns:
    Dictionary containing examples and labels for 'cat', 'dog', and 'combined'.
  """
  validation_ds = tfds.load(
      'cats_vs_dogs',
      split='train[:5%]',
      as_supervised=True,  # Include labels
  )
  validation_ds_resized = validation_ds.map(
      lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y))
  validation_ds_performant = validation_ds_resized.cache().batch(
      BATCH_SIZE).prefetch(buffer_size=10)
  validation_numpy = iter(tfds.as_numpy(validation_ds_performant))

  # Each batch is 32 examples.
  # We create a validation set of 320 examples by taking the first ten batches.
  validation_data = {
      'combined': {
          'examples': [],
          'labels': []
      },
      'cat': {
          'examples': [],
          'labels': []
      },
      'dog': {
          'examples': [],
          'labels': []
      },
  }
  for _ in range(NUM_BATCHES):
    examples, labels = next(validation_numpy)
    for example, label in zip(examples, labels):
      validation_data['combined']['examples'].append(example)
      validation_data['combined']['labels'].append(label)
      if label == 1:
        validation_data['cat']['examples'].append(example)
        validation_data['cat']['labels'].append(label)
      elif label == 0:
        validation_data['dog']['examples'].append(example)
        validation_data['dog']['labels'].append(label)

  return validation_data
