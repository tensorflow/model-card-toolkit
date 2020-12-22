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
"""Model Card Validation.

This submodule contains functions used to validate Python dictionaries against
the Model Card schema.
"""

import json
import os
import pkgutil
from typing import Any, Dict, Text
import jsonschema
import semantic_version

_SCHEMA_FILE_NAME = 'model_card.schema.json'
_SCHEMA_VERSIONS = frozenset(('0.0.1',))
_LATEST_SCHEMA_VERSION = max(_SCHEMA_VERSIONS, key=semantic_version.Version)


def validate_json_schema(model_card_json: Dict[Text, Any],
                         schema_version: Text = _LATEST_SCHEMA_VERSION) -> None:
  """Validates the model card json.

  If schema_version is not provided, it will use the latest schema version.
  See
  https://github.com/tensorflow/model-card-toolkit/tree/master/model_card_toolkit/schema/.

  Args:
    model_card_json: A dictionary following the model card schema.
    schema_version: The version of the model card schema.

  Raises:
    ValueError: If `schema_version` does not correspond to a model card schema
      version.
    ValidationError: If `model_card_json` does not follow the model card schema.
  """
  schema = _find_json_schema(schema_version)
  jsonschema.validate(model_card_json, schema)


def _find_json_schema(schema_version: Text = None) -> Dict[Text, Any]:
  """Returns the model card JSON schema in dictionary format.

  Args:
    schema_version: The version of the schema to fetch. By default, use the
      latest version.

  Returns:
    JSON schema as a dictionary.

  Raises:
    ValueError: If `schema_version` does not correspond to a model card schema
    version.
  """
  if not schema_version:
    schema_version = _LATEST_SCHEMA_VERSION
  if schema_version not in _SCHEMA_VERSIONS:
    raise ValueError(
        'Cannot find schema version that matches the version of the given '
        'model card. Found Versions: {}. Given Version: {}'.format(
            ', '.join(_SCHEMA_VERSIONS), schema_version))

  schema_file = os.path.join('schema', 'v' + schema_version, _SCHEMA_FILE_NAME)
  json_file = pkgutil.get_data('model_card_toolkit', schema_file)
  schema = json.loads(json_file)
  return schema


def get_latest_schema_version() -> Text:
  """Returns the most recent schema version."""
  return _LATEST_SCHEMA_VERSION
