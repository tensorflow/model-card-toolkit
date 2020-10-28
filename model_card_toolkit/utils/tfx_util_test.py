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
"""Tests for the TFX-OSS pipeline utilities."""

import os
import uuid

from absl.testing import absltest

from model_card_toolkit.utils import tfx_util
from model_card_toolkit.utils.testdata import testdata_utils
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class TfxUtilsTest(absltest.TestCase):

  def setUp(self):
    super(TfxUtilsTest, self).setUp()
    self.tmp_db_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    f'test_mlmd_{uuid.uuid4()}.db')

  def _get_empty_metadata_store(self):
    """Returns an empty in memory mlmd store."""
    empty_db_config = metadata_store_pb2.ConnectionConfig()
    empty_db_config.fake_database.SetInParent()
    return mlmd.MetadataStore(empty_db_config)

  def test_get_metrics_artifacts_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    got_metrics = tfx_util.get_metrics_artifacts_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)
    got_metrics_ids = [a.id for a in got_metrics]
    self.assertCountEqual(got_metrics_ids,
                          testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS)

  def test_get_metrics_artifacts_for_model_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.get_metrics_artifacts_for_model(store, model.id)

  def test_get_metrics_artifacts_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.get_metrics_artifacts_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID)

  def test_get_metrics_artifacts_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.get_metrics_artifacts_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)

  def test_get_stats_artifacts_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    got_stats = tfx_util.get_stats_artifacts_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)
    got_stats_ids = [a.id for a in got_stats]
    self.assertCountEqual(got_stats_ids,
                          [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID])

  def test_get_stats_artifacts_for_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.get_stats_artifacts_for_model(store, model.id)

  def test_get_stats_artifacts_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.get_stats_artifacts_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID)

  def test_get_stats_artifacts_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.get_stats_artifacts_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)

  def test_generate_model_card_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    model_card = tfx_util.generate_model_card_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)
    trainers = store.get_executions_by_id([testdata_utils.TFX_0_21_TRAINER_ID])
    self.assertNotEmpty(trainers)
    model_details = model_card.model_details
    self.assertEqual(model_details.name,
                     trainers[-1].properties['module_file'].string_value)
    self.assertEqual(model_details.version.name,
                     trainers[-1].properties['checksum_md5'].string_value)
    self.assertIn(trainers[-1].properties['pipeline_name'].string_value,
                  model_details.references)

    datasets = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_MODEL_DATASET_ID])
    self.assertNotEmpty(datasets)
    model_params = model_card.model_parameters
    self.assertStartsWith(model_params.data.train.name, datasets[-1].uri)
    self.assertStartsWith(model_params.data.eval.name, datasets[-1].uri)

  def test_generate_model_card_for_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.generate_model_card_for_model(store, model.id)

  def test_generate_model_card_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.generate_model_card_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID)

  def test_generate_model_card_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.generate_model_card_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID)

  def test_read_stats_proto(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID])
    self.assertLen(stats, 1)
    train_stats = tfx_util.read_stats_proto(stats[-1].uri, 'train')
    self.assertIsNotNone(train_stats)
    eval_stats = tfx_util.read_stats_proto(stats[-1].uri, 'eval')
    self.assertIsNotNone(eval_stats)

  def test_read_stats_proto_with_invalid_split(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID])
    self.assertLen(stats, 1)
    actual_stats = tfx_util.read_stats_proto(stats[-1].uri, 'invalid_split')
    self.assertIsNone(actual_stats)

  def test_read_stats_proto_with_invalid_uri(self):
    self.assertIsNone(tfx_util.read_stats_proto('/does/not/exist/', 'train'))
    self.assertIsNone(tfx_util.read_stats_proto('/does/not/exist/', 'eval'))

  def test_read_metrics_eval_result(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    metrics = store.get_artifacts_by_id(
        testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS)
    eval_result = tfx_util.read_metrics_eval_result(metrics[-1].uri)
    self.assertIsNotNone(eval_result)

  def test_read_metrics_eval_result_with_invalid_uri(self):
    self.assertIsNone(tfx_util.read_metrics_eval_result('/does/not/exist/'))


if __name__ == '__main__':
  absltest.main()
