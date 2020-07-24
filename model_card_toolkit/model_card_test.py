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
"""Tests for model_card."""

import json
import os

from absl.testing import absltest
import jsonschema

from model_card_toolkit.model_card import Dataset
from model_card_toolkit.model_card import ModelCard
from model_card_toolkit.model_card import Owner

_SCHEMA_DIR = os.path.join(os.path.dirname(__file__), 'schema')
_SCHEMA_FILE = 'model_card.schema.json'


class ModelCardTest(absltest.TestCase):

  def _validate_schema(self, model_card: ModelCard) -> None:
    """Validates the model_card against the json schema.

    Args:
      model_card: The model card data object.

    Raises:
       jsonschema.ValidationError: when the given model_card is
         invalid w.r.t. the schema.
    """
    path = model_card.schema_version if model_card.schema_version else '0.0.1'
    schema_file = os.path.join(_SCHEMA_DIR, 'v' + path, _SCHEMA_FILE)
    with open(schema_file) as json_file:
      schema = json.loads(json_file.read())
    jsonschema.validate(model_card.to_dict(), schema)

  def test_empty_model_card_is_valid_json(self):
    model_card = ModelCard()
    self._validate_schema(model_card)

  def _fill_model_details(self, model_card: ModelCard) -> None:
    """Fills the model_details of the model card."""
    model_details = model_card.model_details
    model_details.name = 'my model'
    model_details.owners = [
        Owner(name='foo', contact='foo@xyz.com'),
        {
            'name': 'bar',
            'contact': 'bar@xyz.com'
        },
    ]
    model_details.version.name = '0.1'
    model_details.version.date = '2020-01-01'
    model_details.license = 'Apache 2.0'
    model_details.references = ['https://my_model.xyz.com']
    model_details.citation = 'https://doi.org/foo/bar'

  def test_model_card_with_model_details_is_valid_json(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_model_details(model_card)
    self._validate_schema(model_card)

  def _fill_model_parameters(self, model_card: ModelCard) -> None:
    """Fills the model_parameters of the model card."""
    model_parameters = model_card.model_parameters
    model_parameters.model_architecture = 'knn'
    model_parameters.data.train.name = 'train_split'
    model_parameters.data.train.link = 'path/to/train'
    model_parameters.data.train.sensitive = False
    model_parameters.data.train.graphics.collection.append({
        'name': 'image1',
        'image': 'rawbytes'
    })
    model_parameters.data.eval = Dataset(name='eval_split', link='path/to/eval')

  def test_model_card_with_model_parameters_is_valid_json(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_model_parameters(model_card)
    self._validate_schema(model_card)

  def _fill_quantitative_analysis(self, model_card: ModelCard) -> None:
    """Fills the quantitative_analysis section of the metadata card."""
    model_card.quantitative_analysis.graphics.collection.append({
        'name': 'image1',
        'image': 'rawbytes'
    })
    model_card.quantitative_analysis.performance_metrics.append({
        'type': 'log_loss',
        'value': 0.2,
    })

  def test_model_card_with_quantitative_analysis_is_valid_json(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_quantitative_analysis(model_card)
    self._validate_schema(model_card)

  def _fill_considerations(self, model_card: ModelCard) -> None:
    """Fills the considerations section of the metadata card."""
    model_card.considerations.users = ['foo', 'bar']
    model_card.considerations.use_cases.append('use case 1')
    model_card.considerations.limitations.append('a limitation')
    model_card.considerations.tradeoffs.append('tradeoff 1')
    model_card.considerations.ethical_considerations.append({
        'name': 'risk1',
        'mitigation_strategy': 'a solution'
    })

  def test_model_card_with_considerations_is_valid_json(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_considerations(model_card)
    self._validate_schema(model_card)

  def test_full_filled_model_card_is_valid_json(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_model_details(model_card)
    self._fill_model_parameters(model_card)
    self._fill_quantitative_analysis(model_card)
    self._fill_considerations(model_card)
    self._validate_schema(model_card)

  def test_default_value_not_shared_among_model_cards(self):
    model_card = ModelCard()
    model_card.schema_version = '0.0.1'
    self._fill_model_details(model_card)
    self._fill_model_parameters(model_card)
    self._fill_quantitative_analysis(model_card)
    self._fill_considerations(model_card)
    other_model_card = ModelCard()
    self.assertNotEqual(other_model_card, model_card)
    self.assertEqual(other_model_card, ModelCard())


if __name__ == '__main__':
  absltest.main()
