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
"""Tests for testdata_utils."""

import os

from absl.testing import absltest

from model_card_toolkit.utils.testdata import testdata_utils


class TestDataUtilsTest(absltest.TestCase):

  def setUp(self):
    super(TestDataUtilsTest, self).setUp()
    tmp_db_path = os.path.join(absltest.get_default_test_tmpdir(), 'test.db')
    self.store = testdata_utils.get_tfx_pipeline_metadata_store(tmp_db_path)
    self.assertIsNotNone(self.store)

  def test_get_tfx_pipeline_metadata_store_model_uri_exists(self):
    self.assertNotEmpty(
        self.store.get_artifacts_by_uri(testdata_utils.TFX_0_21_MODEL_URI))

  def test_get_tfx_pipeline_metadata_store_stats_artifact_exists(self):
    stats = self.store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID])
    self.assertLen(stats, 1)
    self.assertTrue(os.path.exists(stats[-1].uri))

  def test_get_tfx_pipeline_metadata_store_metrics_artifact_exists(self):
    metrics = self.store.get_artifacts_by_id(
        testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS)
    self.assertLen(metrics, len(testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS))
    for artifact in metrics:
      self.assertTrue(os.path.exists(artifact.uri))


if __name__ == '__main__':
  absltest.main()
