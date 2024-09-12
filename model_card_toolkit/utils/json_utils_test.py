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
"""Tests for model_card_toolkit.utils.json_utils."""

import json
import os
import pkgutil

import jsonschema
from absl.testing import absltest, parameterized

from model_card_toolkit.utils import json_utils

_MODEL_DETAILS_V1_DICT = {
    'name':
    'my model',
    'owners': [
        {
            'name': 'foo',
            'contact': 'foo@xyz.com'
        }, {
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }
    ],
    'version': {
        'name': '0.01',
        'date': '2020-01-01'
    },
    'license':
    'Apache 2.0',
    'references': ['https://my_model.xyz.com'],
    'citation':
    'https://doi.org/foo/bar'
}
_MODEL_PARAMETERS_V1_DICT = {
    'model_architecture': 'knn',
    'data': {
        'train': {
            'name': 'train_split',
            'link': 'path/to/train',
            'sensitive': True,
            'graphics': {
                'collection': [{
                    'name': 'image1',
                    'image': 'rawbytes'
                }]
            }
        },
        'eval': {
            'name': 'eval_split',
            'link': 'path/to/eval'
        }
    }
}
_QUANTITATIVE_ANALYSIS_V1_DICT = {
    'graphics': {
        'collection': [{
            'name': 'image1',
            'image': 'rawbytes'
        }]
    },
    'performance_metrics': [{
        'type': 'log_loss',
        'value': 0.2
    }]
}
_CONSIDERATIONS_V1_DICT = {
    'users': ['foo', 'bar'],
    'use_cases': ['use case 1'],
    'limitations': ['a limitation'],
    'tradeoffs': ['tradeoff 1'],
    'ethical_considerations':
    [{
        'name': 'risk1',
        'mitigation_strategy': 'a solution'
    }]
}
_MODEL_CARD_V1_DICT = {
    'model_details': _MODEL_DETAILS_V1_DICT,
    'model_parameters': _MODEL_PARAMETERS_V1_DICT,
    'quantitative_analysis': _QUANTITATIVE_ANALYSIS_V1_DICT,
    'considerations': _CONSIDERATIONS_V1_DICT
}

_MODEL_DETAILS_V2_DICT = {
    'name':
    'my model',
    'owners': [
        {
            'name': 'foo',
            'contact': 'foo@xyz.com'
        }, {
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }
    ],
    'version': {
        'name': '0.01',
        'date': '2020-01-01'
    },
    'license': [{
        'identifier': 'Apache 2.0'
    }],
    'references': [{
        'reference': 'https://my_model.xyz.com'
    }],
    'citation': {
        'citation': 'https://doi.org/foo/bar'
    }
}
_MODEL_PARAMETERS_V2_DICT = {
    'model_architecture':
    'knn',
    'data': [
        {
            'name': 'train_split',
            'link': 'path/to/train',
            'sensitive': {
                'sensitive_data': [
                    'this dataset contains PII',
                    'this dataset contains geo data'
                ]
            },
            'graphics': {
                'collection': [{
                    'name': 'image1',
                    'image': 'rawbytes'
                }]
            }
        }, {
            'name': 'eval_split',
            'link': 'path/to/eval'
        }
    ]
}
_QUANTITATIVE_ANALYSIS_V2_DICT = {
    'graphics': {
        'collection': [{
            'name': 'image1',
            'image': 'rawbytes'
        }]
    },
    'performance_metrics': [{
        'type': 'log_loss',
        'value': '0.2'
    }]
}
_CONSIDERATIONS_V2_DICT = {
    'users': [{
        'description': 'foo'
    }, {
        'description': 'bar'
    }],
    'use_cases': [{
        'description': 'use case 1'
    }],
    'limitations': [{
        'description': 'a limitation'
    }],
    'tradeoffs': [{
        'description': 'tradeoff 1'
    }],
    'ethical_considerations':
    [{
        'name': 'risk1',
        'mitigation_strategy': 'a solution'
    }]
}
_MODEL_CARD_V2_DICT = {
    'model_details': _MODEL_DETAILS_V2_DICT,
    'model_parameters': _MODEL_PARAMETERS_V2_DICT,
    'quantitative_analysis': _QUANTITATIVE_ANALYSIS_V2_DICT,
    'considerations': _CONSIDERATIONS_V2_DICT
}

_CATS_VS_DOGS_V1_PATH = os.path.join(
    'utils', 'testdata', 'cats_vs_dogs_v0_0_1.json'
)
_CATS_VS_DOGS_V1_TEXT = pkgutil.get_data(
    'model_card_toolkit', _CATS_VS_DOGS_V1_PATH
)
_CATS_VS_DOGS_V1_DICT = json.loads(_CATS_VS_DOGS_V1_TEXT)

_CATS_VS_DOGS_V2_PATH = os.path.join(
    'utils', 'testdata', 'cats_vs_dogs_v0_0_2.json'
)
_CATS_VS_DOGS_V2_TEXT = pkgutil.get_data(
    'model_card_toolkit', _CATS_VS_DOGS_V2_PATH
)
_CATS_VS_DOGS_V2_DICT = json.loads(_CATS_VS_DOGS_V2_TEXT)

_MODEL_CARD_SECTIONS = ('model_details', 'model_parameters', 'considerations')
_QUANTITATIVE_ANALYSIS = 'quantitative_analysis'
_CONFIDENCE_INTERVAL = 'confidence_interval'


class JsonUtilsTest(parameterized.TestCase):
  def test_validate_json_schema(self):
    json_utils.validate_json_schema(
        _MODEL_CARD_V1_DICT, schema_version='0.0.1'
    )
    json_utils.validate_json_schema(
        _MODEL_CARD_V2_DICT, schema_version='0.0.2'
    )

  def test_validate_json_schema_invalid_dict(self):
    invalid_json_dict = {'model_name': 'the_greatest_model'}
    with self.assertRaises(jsonschema.ValidationError):
      json_utils.validate_json_schema(invalid_json_dict)

  def test_validate_json_schema_invalid_version(self):
    invalid_schema_version = '0.0.3'
    with self.assertRaises(ValueError):
      json_utils.validate_json_schema(
          _MODEL_CARD_V1_DICT, schema_version=invalid_schema_version
      )

  @parameterized.named_parameters(
      ('train_data', 'train_data.json'),
      ('considerations', 'considerations.json'),
      ('cats_vs_dogs', 'cats_vs_dogs_v0_0_2.json'), ('full', 'full.json')
  )
  def test_template_test_files(self, file_name):
    template_path = os.path.join('utils', 'testdata', file_name)
    json_data = json.loads(
        pkgutil.get_data('model_card_toolkit', template_path)
    )
    json_utils.validate_json_schema(json_data)

  def test_json_update_succeeds(self):

    updated_cats_vs_dogs_dict = json_utils.update(
        json_dict=_CATS_VS_DOGS_V1_DICT
    )

    for section in _MODEL_CARD_SECTIONS:
      with self.subTest(name=section):
        self.assertDictEqual(
            updated_cats_vs_dogs_dict.get(section),
            _CATS_VS_DOGS_V2_DICT.get(section)
        )

    with self.subTest(name=_QUANTITATIVE_ANALYSIS):

      # Check Graphics fields are equal
      v1_graphics_updated = updated_cats_vs_dogs_dict.get(
          _QUANTITATIVE_ANALYSIS
      ).get('graphics')
      v2_graphics = _CATS_VS_DOGS_V2_DICT.get(_QUANTITATIVE_ANALYSIS
                                              ).get('graphics')
      self.assertDictEqual(v1_graphics_updated, v2_graphics)

      # Check PerformanceMetrics fields are equal length
      v1_metrics_updated = updated_cats_vs_dogs_dict.get(
          _QUANTITATIVE_ANALYSIS
      ).get('performance_metrics')
      v2_metrics = _CATS_VS_DOGS_V2_DICT.get(_QUANTITATIVE_ANALYSIS
                                             ).get('performance_metrics')
      self.assertLen(v1_metrics_updated, len(v2_metrics))

      # Check PerformanceMetrics fields are equal value
      for v1m, v2m in zip(v1_metrics_updated, v2_metrics):
        self.assertSameElements(v1m.keys(), v2m.keys())
        for field in v1m.keys():
          if field == _CONFIDENCE_INTERVAL:
            self.assertEqual(
                str(v1m[_CONFIDENCE_INTERVAL]['lower_bound']),
                str(v2m[_CONFIDENCE_INTERVAL]['lower_bound'])
            )
            self.assertEqual(
                str(v1m[_CONFIDENCE_INTERVAL]['upper_bound']),
                str(v2m[_CONFIDENCE_INTERVAL]['upper_bound'])
            )
          else:
            self.assertEqual(str(v1m[field]), str(v2m[field]))

  def test_json_update_latest_version_should_be_identity_function(self):
    updated_cats_vs_dogs_dict = json_utils.update(
        json_dict=_CATS_VS_DOGS_V2_DICT
    )
    for section in _MODEL_CARD_SECTIONS:
      with self.subTest(name=section):
        self.assertDictEqual(
            updated_cats_vs_dogs_dict.get(section),
            _CATS_VS_DOGS_V2_DICT.get(section)
        )

  def test_json_update_validation_error(self):
    with self.assertRaises(jsonschema.ValidationError):
      json_utils.update(json_dict={'model_name': 'the_greatest_model'})


if __name__ == '__main__':
  absltest.main()
