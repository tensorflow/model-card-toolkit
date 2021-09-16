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
"""Tests for model_card_toolkit.model_card."""

import json
import os
import pkgutil
from absl.testing import absltest
import jsonschema

from model_card_toolkit import model_card
from model_card_toolkit.proto import model_card_pb2

from google.protobuf import text_format

_FULL_PROTO_FILE_NAME = "full.pbtxt"
_FULL_PROTO = pkgutil.get_data(
    "model_card_toolkit",
    os.path.join("template/test", _FULL_PROTO_FILE_NAME))
_FULL_JSON_FILE_PATH = "full.json"
_FULL_JSON = model_card_json_bytestring = pkgutil.get_data(
    "model_card_toolkit",
    os.path.join("template/test", _FULL_JSON_FILE_PATH))


class ModelCardTest(absltest.TestCase):

  def test_copy_from_proto_and_to_proto_with_all_fields(self):
    want_proto = text_format.Parse(_FULL_PROTO, model_card_pb2.ModelCard())
    model_card_py = model_card.ModelCard()
    model_card_py.copy_from_proto(want_proto)
    got_proto = model_card_py.to_proto()

    self.assertEqual(want_proto, got_proto)

  def test_merge_from_proto_and_to_proto_with_all_fields(self):
    want_proto = text_format.Parse(_FULL_PROTO, model_card_pb2.ModelCard())
    model_card_py = model_card.ModelCard()
    model_card_py.merge_from_proto(want_proto)
    got_proto = model_card_py.to_proto()

    self.assertEqual(want_proto, got_proto)

  def test_copy_from_proto_sucess(self):
    # Test fields convert.
    owner = model_card.Owner(name="my_name1")
    owner_proto = model_card_pb2.Owner(name="my_name2", contact="my_contact2")
    owner.copy_from_proto(owner_proto)
    self.assertEqual(owner,
                     model_card.Owner(name="my_name2", contact="my_contact2"))

    # Test message convert.
    model_details = model_card.ModelDetails(
        owners=[model_card.Owner(name="my_name1")])
    model_details_proto = model_card_pb2.ModelDetails(
        owners=[model_card_pb2.Owner(name="my_name2", contact="my_contact2")])
    model_details.copy_from_proto(model_details_proto)
    self.assertEqual(
        model_details,
        model_card.ModelDetails(
            owners=[model_card.Owner(name="my_name2", contact="my_contact2")]))

  def test_merge_from_proto_sucess(self):
    # Test fields convert.
    owner = model_card.Owner(name="my_name1")
    owner_proto = model_card_pb2.Owner(contact="my_contact1")
    owner.merge_from_proto(owner_proto)
    self.assertEqual(owner,
                     model_card.Owner(name="my_name1", contact="my_contact1"))

    # Test message convert.
    model_details = model_card.ModelDetails(
        owners=[model_card.Owner(name="my_name1")])
    model_details_proto = model_card_pb2.ModelDetails(
        owners=[model_card_pb2.Owner(name="my_name2", contact="my_contact2")])
    model_details.merge_from_proto(model_details_proto)
    self.assertEqual(
        model_details,
        model_card.ModelDetails(owners=[
            model_card.Owner(name="my_name1"),
            model_card.Owner(name="my_name2", contact="my_contact2")
        ]))

  def test_copy_from_proto_with_invalid_proto(self):
    owner = model_card.Owner()
    wrong_proto = model_card_pb2.Version()
    with self.assertRaisesRegex(
        TypeError,
        "<class 'model_card_toolkit.proto.model_card_pb2.Owner'> is expected. "
        "However <class 'model_card_toolkit.proto.model_card_pb2.Version'> is "
        "provided."):
      owner.copy_from_proto(wrong_proto)

  def test_merge_from_proto_with_invalid_proto(self):
    owner = model_card.Owner()
    wrong_proto = model_card_pb2.Version()
    with self.assertRaisesRegex(
        TypeError, ".*expected .*Owner got .*Version.*"):
      owner.merge_from_proto(wrong_proto)

  def test_to_proto_sucess(self):
    # Test fields convert.
    owner = model_card.Owner()
    self.assertEqual(owner.to_proto(), model_card_pb2.Owner())
    owner.name = "my_name"
    self.assertEqual(owner.to_proto(), model_card_pb2.Owner(name="my_name"))
    owner.contact = "my_contact"
    self.assertEqual(owner.to_proto(),
                     model_card_pb2.Owner(name="my_name", contact="my_contact"))

    # Test message convert.
    model_details = model_card.ModelDetails(
        owners=[model_card.Owner(name="my_name", contact="my_contact")])
    self.assertEqual(
        model_details.to_proto(),
        model_card_pb2.ModelDetails(
            owners=[model_card_pb2.Owner(name="my_name", contact="my_contact")],
            version=model_card_pb2.Version()))

  def test_to_proto_with_invalid_field(self):
    owner = model_card.Owner()
    owner.wrong_field = "wrong"
    with self.assertRaisesRegex(ValueError,
                                "has no such field named 'wrong_field'."):
      owner.to_proto()

  def test_from_json_and_to_json_with_all_fields(self):
    want_json = json.loads(_FULL_JSON)
    model_card_py = model_card.ModelCard()._from_json(want_json)
    got_json = json.loads(model_card_py.to_json())
    self.assertEqual(want_json, got_json)

  def test_from_json_overwrites_previous_fields(self):
    overwritten_limitation = model_card.Limitation(
        description="This model can only be used on text up to 140 characters.")
    model_card_py = model_card.ModelCard(
        considerations=model_card.Considerations(
            limitations=[overwritten_limitation]))
    model_card_json = json.loads(_FULL_JSON)
    model_card_py = model_card_py._from_json(model_card_json)
    self.assertNotIn(overwritten_limitation,
                     model_card_py.considerations.limitations)

  def test_from_invalid_json(self):
    invalid_json_dict = {"model_name": "the_greatest_model"}
    with self.assertRaises(jsonschema.ValidationError):
      model_card.ModelCard()._from_json(invalid_json_dict)

  def test_from_invalid_json_vesion(self):
    model_card_dict = {
        "model_details": {},
        "model_parameters": {},
        "quantitative_analysis": {},
        "considerations": {},
        "schema_version": "0.0.3"
    }
    with self.assertRaisesRegex(ValueError, (
        "^Cannot find schema version that matches the version of the given "
        "model card.")):
      model_card.ModelCard()._from_json(model_card_dict)

  def test_from_proto_to_json(self):
    model_card_proto = text_format.Parse(_FULL_PROTO,
                                         model_card_pb2.ModelCard())
    model_card_py = model_card.ModelCard()

    # Use merge_from_proto.
    self.assertJsonEqual(
        _FULL_JSON,
        model_card_py.merge_from_proto(model_card_proto).to_json())
    # Use copy_from_proto
    self.assertJsonEqual(
        _FULL_JSON,
        model_card_py.copy_from_proto(model_card_proto).to_json())

  def test_from_json_to_proto(self):
    model_card_proto = text_format.Parse(_FULL_PROTO,
                                         model_card_pb2.ModelCard())

    model_card_json = json.loads(_FULL_JSON)
    model_card_py = model_card.ModelCard()._from_json(model_card_json)
    model_card_json2proto = model_card_py.to_proto()

    self.assertEqual(model_card_proto, model_card_json2proto)


if __name__ == "__main__":
  absltest.main()
