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
"""Tests for model_card_toolkit.validation."""

from absl.testing import absltest
import jsonschema
from model_card_toolkit.utils import validation

_MODEL_DETAILS_DICT = {
    "name": "my model",
    "owners": [{
        "name": "foo",
        "contact": "foo@xyz.com"
    }, {
        "name": "bar",
        "contact": "bar@xyz.com"
    }],
    "version": {
        "name": "0.01",
        "date": "2020-01-01"
    },
    "license": "Apache 2.0",
    "references": ["https://my_model.xyz.com"],
    "citation": "https://doi.org/foo/bar"
}
_MODEL_PARAMETERS_DICT = {
    "model_architecture": "knn",
    "data": {
        "train": {
            "name": "train_split",
            "link": "path/to/train",
            "sensitive": True,
            "graphics": {
                "collection": [{
                    "name": "image1",
                    "image": "rawbytes"
                }]
            }
        },
        "eval": {
            "name": "eval_split",
            "link": "path/to/eval"
        }
    }
}
_QUANTITATIVE_ANALYSIS_DICT = {
    "graphics": {
        "collection": [{
            "name": "image1",
            "image": "rawbytes"
        }]
    },
    "performance_metrics": [{
        "type": "log_loss",
        "value": 0.2
    }]
}
_CONSIDERATIONS_DICT = {
    "users": ["foo", "bar"],
    "use_cases": ["use case 1"],
    "limitations": ["a limitation"],
    "tradeoffs": ["tradeoff 1"],
    "ethical_considerations": [{
        "name": "risk1",
        "mitigation_strategy": "a solution"
    }]
}
_MODEL_CARD_DICT = {
    "model_details": _MODEL_DETAILS_DICT,
    "model_parameters": _MODEL_PARAMETERS_DICT,
    "quantitative_analysis": _QUANTITATIVE_ANALYSIS_DICT,
    "considerations": _CONSIDERATIONS_DICT
}
_SCHEMA_VERSION = "0.0.1"


class ValidationTest(absltest.TestCase):

  def test_validate_json_schema(self):
    validation.validate_json_schema(_MODEL_CARD_DICT, _SCHEMA_VERSION)

  def test_validate_json_schema_invalid_dict(self):
    invalid_schema_dict = {"model_name": "the_greatest_model"}
    with self.assertRaises(jsonschema.ValidationError):
      validation.validate_json_schema(invalid_schema_dict)

  def test_validate_json_schema_invalid_version(self):
    invalid_schema_version = "0.0.2"
    with self.assertRaises(ValueError):
      validation.validate_json_schema(_MODEL_CARD_DICT, invalid_schema_version)


if __name__ == "__main__":
  absltest.main()
