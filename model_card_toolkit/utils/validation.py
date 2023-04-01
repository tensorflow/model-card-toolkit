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
from typing import Any, Dict, Optional

import jsonschema

_SCHEMA_FILE_NAME = 'model_card.schema.json'
_SCHEMA_VERSIONS = frozenset((
    '0.0.1',
    '0.0.2',
))
_LATEST_SCHEMA_VERSION = '0.0.2'

SCHEMA_VERSION_STRING = 'schema_version'


def validate_json_schema(
    json_dict: Dict[str, Any], schema_version: Optional[str] = None
) -> Dict[str, Any]:
  """Validates the json schema of a model card field.

  If schema_version is not provided, it will use the latest schema version.
  See
  https://github.com/tensorflow/model-card-toolkit/tree/main/model_card_toolkit/schema/.

  Args:
    json_dict: A dictionary following the schema for a model card field.
    schema_version: The version of the model card schema. Optional field; if
      omitted, defers to the latest schema version.

  Returns:
    The schema used for validation.

  Raises:
    ValueError: If `schema_version` does not correspond to a model card schema
      version.
    ValidationError: If `model_card_json` does not follow the model card schema.
  """
  schema = _find_json_schema(
      schema_version or json_dict.get('schema_version')
      or _LATEST_SCHEMA_VERSION
  )
  jsonschema.validate(json_dict, schema)
  return schema


def _find_json_schema(schema_version: Optional[str] = None) -> Dict[str, Any]:
  """Returns the JSON schema of a model card field in dictionary format.

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
            ', '.join(_SCHEMA_VERSIONS), schema_version
        )
    )

  schema_file = os.path.join('schema', 'v' + schema_version, _SCHEMA_FILE_NAME)
  json_file = pkgutil.get_data('model_card_toolkit', schema_file)
  schema = json.loads(json_file)
  return schema


def get_latest_schema_version() -> str:
  """Returns the most recent schema version."""
  return _LATEST_SCHEMA_VERSION
