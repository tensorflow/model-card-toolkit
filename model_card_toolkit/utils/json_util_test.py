"""Tests for model_card_toolkit.utils.json_util."""

import json
import os
import pkgutil
from absl.testing import absltest
import jsonschema
from model_card_toolkit.utils import json_util

_CATS_VS_DOGS_V1_PATH = os.path.join("utils", "testdata", "cats_vs_dogs.json")
_CATS_VS_DOGS_V1_TEXT = pkgutil.get_data("model_card_toolkit",
                                         _CATS_VS_DOGS_V1_PATH)
_CATS_VS_DOGS_V1_DICT = json.loads(_CATS_VS_DOGS_V1_TEXT)

_CATS_VS_DOGS_V2_PATH = os.path.join("template", "test", "cats_vs_dogs.json")
_CATS_VS_DOGS_V2_TEXT = pkgutil.get_data("model_card_toolkit",
                                         _CATS_VS_DOGS_V2_PATH)
_CATS_VS_DOGS_V2_DICT = json.loads(_CATS_VS_DOGS_V2_TEXT)

_DICT_KEYS = ("model_details", "model_parameters", "quantitative_analysis",
              "considerations")


class JsonUtilTest(absltest.TestCase):

  def test_json_update_succeeds(self):
    updated_cats_vs_dogs_dict = json_util.update(
        json_dict=_CATS_VS_DOGS_V1_DICT)
    for k in _DICT_KEYS:
      with self.subTest(name=k):
        self.assertDictEqual(
            updated_cats_vs_dogs_dict.get(k), _CATS_VS_DOGS_V2_DICT.get(k))

  def test_json_update_latest_version_should_be_identity_function(self):
    updated_cats_vs_dogs_dict = json_util.update(
        json_dict=_CATS_VS_DOGS_V2_DICT)
    for k in _DICT_KEYS:
      with self.subTest(name=k):
        self.assertDictEqual(
            updated_cats_vs_dogs_dict.get(k), _CATS_VS_DOGS_V2_DICT.get(k))

  def test_json_update_validation_error(self):
    with self.assertRaises(jsonschema.ValidationError):
      json_util.update(json_dict={"model_name": "the_greatest_model"})


if __name__ == "__main__":
  absltest.main()
