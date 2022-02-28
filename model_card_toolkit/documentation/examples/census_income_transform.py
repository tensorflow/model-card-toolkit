"""Census Income utilities.

A collection of helper functions for the Census Income classifier used in
MLMD_Model_Card_Toolkit_Demo.ipynb.
"""

from model_card_toolkit.documentation.examples import census_income_constants
import tensorflow as tf
import tensorflow_transform as tft

_DENSE_FLOAT_FEATURE_KEYS = census_income_constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = census_income_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = census_income_constants.VOCAB_SIZE
_OOV_SIZE = census_income_constants.OOV_SIZE
_FEATURE_BUCKET_COUNT = census_income_constants.FEATURE_BUCKET_COUNT
_BUCKET_FEATURE_KEYS = census_income_constants.BUCKET_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = census_income_constants.CATEGORICAL_FEATURE_KEYS
_LABEL_KEY = census_income_constants.LABEL_KEY
_transformed_name = census_income_constants.transformed_name


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[_transformed_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]),
        top_k=_VOCAB_SIZE,
        num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = tft.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  label = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = label

  return outputs


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
