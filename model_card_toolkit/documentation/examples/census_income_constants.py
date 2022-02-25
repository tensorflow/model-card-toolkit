"""Census Income utilities.

A collection of constants to be used by the Census Income Classifier.
"""

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [20]

CATEGORICAL_FEATURE_KEYS = ["Education-Num"]


DENSE_FLOAT_FEATURE_KEYS = ["Capital-Gain", "Hours-per-week", "Capital-Loss"]

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = ["Age"]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 200

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = ["Workclass", "Education", "Marital-Status", "Occupation",
                      "Relationship", "Race", "Sex", "Country"]

# Keys
LABEL_KEY = "Over-50K"


def transformed_name(key):
  return key + "_xf"
