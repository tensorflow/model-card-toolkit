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
"""Util functions for Model Card JSON schema."""

import logging
from typing import Any, Dict, Optional, Text

import jsonschema

from model_card_toolkit.utils import validation


def update(json_dict: Optional[Dict[Text, Any]] = None) -> Dict[Text, Any]:
  """Updates a Model Card JSON dictionary to the latest schema version.

  If you have a JSON string, you can use it with this function as follows:

  ```python
  json_dict = json.loads(json_text)
  updated_json_dict = json_util.update(json_dict)
  ```

  Args:
    json_dict: A dictionary representing a Model Card JSON object.

  Returns:
    The input Model Card, converted to a JSON string of the latest schema
      version. If the input Model Card already corresponds to schema v0.0.2, it
      is returned unmodified.

  Raises:
    ValidationError: If `json_dict` does not follow the model card JSON v0.0.1
      schema.
  """
  try:
    validation.validate_json_schema(json_dict, "0.0.2")
    logging.info("JSON object already matches schema 0.0.2.")
    return json_dict  # pytype: disable=bad-return-type
  except jsonschema.ValidationError:
    logging.info("JSON object does match schema 0.0.2; updating.")
    return _update_from_v1_to_v2(json_dict)


def _update_from_v1_to_v2(json_dict: Dict[Text, Any]) -> Dict[Text, Any]:
  """Updates a Model Card JSON v0.0.1 string to v0.0.2.

  Args:
    json_dict: A dictionary representing a Model Card v0.0.1 JSON object.

  Returns:
    The input Model Card, converted to a v0.0.2 JSON string.

  Raises:
      JSONDecodeError: If `json_dict` is not a valid JSON string.
      ValidationError: If `json_dict` does not follow the model card JSON v0.0.1
        schema.
  """

  # Validate input args schema
  validation.validate_json_schema(json_dict, "0.0.1")

  # Update schema version
  json_dict["schema_version"] = validation.get_latest_schema_version()

  # Update model_details
  if json_dict["model_details"]["license"]:
    json_dict["model_details"]["licenses"] = [
        {
            "custom_text": json_dict["model_details"].pop("license")
        }
    ]
  if json_dict["model_details"]["references"]:
    json_dict["model_details"]["references"] = [
        {
            "reference": reference
        } for reference in json_dict["model_details"]["references"]
    ]
  if json_dict["model_details"]["citation"]:
    json_dict["model_details"]["citations"] = [
        {
            "citation": json_dict["model_details"].pop("citation")
        }
    ]

  # Update model_parameters
  if "model_parameters" in json_dict and "data" in json_dict["model_parameters"
                                                             ]:
    new_data = []
    if "train" in json_dict["model_parameters"]["data"]:
      old_train_data = json_dict["model_parameters"]["data"]["train"]
      if "name" not in old_train_data:
        old_train_data["name"] = "Training Set"
      new_data.append(old_train_data)
    if "eval" in json_dict["model_parameters"]["data"]:
      old_eval_data = json_dict["model_parameters"]["data"]["eval"]
      if "name" not in old_eval_data:
        old_eval_data["name"] = "Validation Set"
      new_data.append(old_eval_data)
    json_dict["model_parameters"]["data"] = new_data

  # Update considerations
  if "considerations" in json_dict and "use_cases" in json_dict[
      "considerations"]:
    json_dict["considerations"]["use_cases"] = [
        {
            "description": use_case
        } for use_case in json_dict["considerations"]["use_cases"]
    ]
  if "considerations" in json_dict and "users" in json_dict["considerations"]:
    json_dict["considerations"]["users"] = [
        {
            "description": user
        } for user in json_dict["considerations"]["users"]
    ]
  if "considerations" in json_dict and "limitations" in json_dict[
      "considerations"]:
    json_dict["considerations"]["limitations"] = [
        {
            "description": limitation
        } for limitation in json_dict["considerations"]["limitations"]
    ]
  if "considerations" in json_dict and "tradeoffs" in json_dict[
      "considerations"]:
    json_dict["considerations"]["tradeoffs"] = [
        {
            "description": limitation
        } for limitation in json_dict["considerations"]["tradeoffs"]
    ]

  return json_dict
