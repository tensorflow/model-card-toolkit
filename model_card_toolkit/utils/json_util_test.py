"""Tests for model_card_toolkit.utils.json_util."""

import json
import os
import pkgutil

import jsonschema
from absl.testing import absltest

from model_card_toolkit.utils import json_util

_CATS_VS_DOGS_V1_PATH = os.path.join(
    "utils", "testdata", "cats_vs_dogs_v0_0_1.json"
)
_CATS_VS_DOGS_V1_TEXT = pkgutil.get_data(
    "model_card_toolkit", _CATS_VS_DOGS_V1_PATH
)
_CATS_VS_DOGS_V1_DICT = json.loads(_CATS_VS_DOGS_V1_TEXT)

_CATS_VS_DOGS_V2_PATH = os.path.join(
    "utils", "testdata", "cats_vs_dogs_v0_0_2.json"
)
_CATS_VS_DOGS_V2_TEXT = pkgutil.get_data(
    "model_card_toolkit", _CATS_VS_DOGS_V2_PATH
)
_CATS_VS_DOGS_V2_DICT = json.loads(_CATS_VS_DOGS_V2_TEXT)

_MODEL_CARD_SECTIONS = ("model_details", "model_parameters", "considerations")
_QUANTITATIVE_ANALYSIS = "quantitative_analysis"
_CONFIDENCE_INTERVAL = "confidence_interval"


class JsonUtilTest(absltest.TestCase):
  def test_json_update_succeeds(self):

    updated_cats_vs_dogs_dict = json_util.update(
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
      ).get("graphics")
      v2_graphics = _CATS_VS_DOGS_V2_DICT.get(_QUANTITATIVE_ANALYSIS
                                              ).get("graphics")
      self.assertDictEqual(v1_graphics_updated, v2_graphics)

      # Check PerformanceMetrics fields are equal length
      v1_metrics_updated = updated_cats_vs_dogs_dict.get(
          _QUANTITATIVE_ANALYSIS
      ).get("performance_metrics")
      v2_metrics = _CATS_VS_DOGS_V2_DICT.get(_QUANTITATIVE_ANALYSIS
                                             ).get("performance_metrics")
      self.assertLen(v1_metrics_updated, len(v2_metrics))

      # Check PerformanceMetrics fields are equal value
      for v1m, v2m in zip(v1_metrics_updated, v2_metrics):
        self.assertSameElements(v1m.keys(), v2m.keys())
        for field in v1m.keys():
          if field == _CONFIDENCE_INTERVAL:
            self.assertEqual(
                str(v1m[_CONFIDENCE_INTERVAL]["lower_bound"]),
                str(v2m[_CONFIDENCE_INTERVAL]["lower_bound"])
            )
            self.assertEqual(
                str(v1m[_CONFIDENCE_INTERVAL]["upper_bound"]),
                str(v2m[_CONFIDENCE_INTERVAL]["upper_bound"])
            )
          else:
            self.assertEqual(str(v1m[field]), str(v2m[field]))

  def test_json_update_latest_version_should_be_identity_function(self):
    updated_cats_vs_dogs_dict = json_util.update(
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
      json_util.update(json_dict={"model_name": "the_greatest_model"})


if __name__ == "__main__":
  absltest.main()
