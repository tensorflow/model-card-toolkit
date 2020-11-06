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
DEFAULT_TRAINING_EPOCHS = 4


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


def create_model(
    training_epochs: int = DEFAULT_TRAINING_EPOCHS) -> tf.keras.Model:
  """Create and train model used in Standalone Model Card Toolkit notebook.

  This is a MobileNetV2-architecture model, using pretrained weights based on
  ImageNet. In this function, the model weights are further trained on the
  Cats vs Dogs dataset.

  This model is based on the model from
  https://www.tensorflow.org/guide/keras/transfer_learning.

  This model is used in
  https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/Standalone_Model_Card_Toolkit_Demo.ipynb.

  Args:
    training_epochs: The number of epochs to train the model over. 4 by default.

  Returns:
    Model used in Standalone Model Card Toolkit notebook.
  """
  resize = lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y)

  train_ds, validation_ds = tfds.load(
      'cats_vs_dogs',
      split=['train[:20%]', 'train[20%:25%]'],
      as_supervised=True,  # Include labels
  )
  train_ds = train_ds.map(resize).cache().batch(BATCH_SIZE).prefetch(
      buffer_size=10)
  validation_ds = validation_ds.map(resize).cache().batch(BATCH_SIZE).prefetch(
      buffer_size=10)

  base_model = tf.keras.applications.MobileNetV2(
      weights='imagenet',  # Load weights pre-trained on ImageNet.
      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
      include_top=False,
  )  # Do not include the ImageNet classifier at the top.

  # Freeze the convolutional base of the MobileNetV2 model
  base_model.trainable = False

  # Create new model on top
  inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  ])
  x = data_augmentation(inputs)  # Apply random data augmentation
  x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

  x = base_model(x, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  outputs = tf.keras.layers.Dense(1)(x)
  model = tf.keras.Model(inputs, outputs)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.BinaryAccuracy()])

  model.fit(
      train_ds,
      epochs=training_epochs,
      validation_data=validation_ds,
  )

  return model
