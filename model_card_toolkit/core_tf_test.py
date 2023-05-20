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
"""TensorFlow tests for model_card_toolkit.core."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest, parameterized

try:
  import tensorflow_model_analysis as tfma
  from ml_metadata.proto import metadata_store_pb2
except ImportError:
  pass

from model_card_toolkit import core, dependencies, model_card

try:
  from model_card_toolkit.utils import tf_graphics, tf_sources
  from model_card_toolkit.utils.testdata import tf_testdata_utils
  from model_card_toolkit.utils.testdata.tfxtest import TfxTest
  from model_card_toolkit.utils.tf_utils import (
      _TFX_METRICS_TYPE, _TFX_STATS_TYPE
  )
except ImportError:
  tf_graphics = None
  TfxTest = absltest.TestCase

_MOCK_TENSORFLOW_EXTRA_MISSING_DEP = {
    dependencies._TENSORFLOW_EXTRA_DEPS[0]: None,
}

_IS_MISSING_OPTIONAL_DEPS = not dependencies.has_tensorflow_extra_deps()


class TfCoreTest(parameterized.TestCase, TfxTest):
  def setUp(self):
    super().setUp()
    if _IS_MISSING_OPTIONAL_DEPS:
      self.skipTest('Missing optional dependencies.')
    test_dir = self.create_tempdir()
    self.tmp_db_path = os.path.join(test_dir, 'test_mlmd.db')
    self.mct_dir = test_dir.mkdir(
        os.path.join(test_dir, 'model_card')
    ).full_path

  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MISSING_DEP)
  def test_init_with_store_and_missing_tensorflow_extra_deps(self):
    store = tf_testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaises(ImportError):
      core.ModelCardToolkit(
          output_dir=self.mct_dir, mlmd_source=tf_sources.MlmdSource(
              store=store, model_uri=tf_testdata_utils.TFX_0_21_MODEL_URI
          )
      )

  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MISSING_DEP)
  def test_init_with_source_and_missing_tensorflow_extra_deps(self):
    with self.assertRaises(ImportError):
      core.ModelCardToolkit(source=tf_sources.Source())

  def test_init_with_store_model_uri_not_found(self):
    store = tf_testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    unknown_model = 'unknown_model'
    with self.assertRaisesRegex(
        ValueError, f'"{unknown_model}" cannot be found in the `store`'
    ):
      core.ModelCardToolkit(
          mlmd_source=tf_sources.MlmdSource(
              store=store, model_uri=unknown_model
          )
      )  # yapf: disable

  @mock.patch.object(
      tf_graphics, 'annotate_dataset_feature_statistics_plots', autospec=True
  )
  @mock.patch.object(tf_graphics, 'annotate_eval_result_plots', autospec=True)
  def test_scaffold_assets_with_store(
      self, mock_annotate_data_stats, mock_annotate_eval_results
  ):
    num_stat_artifacts = 2
    num_eval_artifacts = 1
    output_dir = self.mct_dir
    store = tf_testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = core.ModelCardToolkit(
        output_dir=output_dir, mlmd_source=tf_sources.MlmdSource(
            store=store, model_uri=tf_testdata_utils.TFX_0_21_MODEL_URI
        )
    )
    mc = mct.scaffold_assets()
    self.assertIsNotNone(mc.model_details.name)
    self.assertIsNotNone(mc.model_details.version.name)
    self.assertIn(
        'default_template.html.jinja',
        os.listdir(os.path.join(output_dir, 'template/html'))
    )
    self.assertIn(
        'default_template.md.jinja',
        os.listdir(os.path.join(output_dir, 'template/md'))
    )
    self.assertEqual(mock_annotate_data_stats.call_count, num_stat_artifacts)
    self.assertEqual(mock_annotate_eval_results.call_count, num_eval_artifacts)

  @parameterized.parameters(
      ('', True), ('', False), ('tfrecord', True), ('tfrecord', False)
  )
  def test_scaffold_assets_with_source(
      self, output_file_format: str, artifacts: bool
  ):

    train_dataset_name = 'Dataset-Split-train'
    train_features = ['feature_name1']
    eval_dataset_name = 'Dataset-Split-eval'
    eval_features = ['feature_name2', 'feature_name3']

    test_dir = self.create_tempdir()
    tfma_path = os.path.join(test_dir, 'tfma')
    tfdv_path = os.path.join(test_dir, 'tfdv')
    pushed_model_path = os.path.join(test_dir, 'pushed_model')

    add_metrics_callbacks = [
        tfma.post_export_metrics.example_count(),
        tfma.post_export_metrics.calibration_plot_and_prediction_histogram(
            num_buckets=2
        ),
    ]

    if artifacts:
      mlmd_store = self._set_up_mlmd()
      self._write_tfma(
          tfma_path, output_file_format, add_metrics_callbacks, mlmd_store
      )
      self._write_tfdv(
          tfdv_path, train_dataset_name, train_features, eval_dataset_name,
          eval_features, mlmd_store
      )
      model_evaluation_artifacts = mlmd_store.get_artifacts_by_type(
          _TFX_METRICS_TYPE
      )
      example_statistics_artifacts = mlmd_store.get_artifacts_by_type(
          _TFX_STATS_TYPE
      )
      # Use placeholder artifact to avoid introducing tfx as a dependency
      pushed_model_artifact = metadata_store_pb2.Artifact(
          uri=pushed_model_path
      )
      tfma_src = tf_sources.TfmaSource(
          model_evaluation_artifacts=model_evaluation_artifacts,
          metrics_exclude=['average_loss']
      )
      tfdv_src = tf_sources.TfdvSource(
          example_statistics_artifacts=example_statistics_artifacts,
          features_include=['feature_name1', 'feature_name3']
      )
      model_src = tf_sources.ModelSource(
          pushed_model_artifact=pushed_model_artifact
      )
    else:
      self._write_tfma(tfma_path, output_file_format, add_metrics_callbacks)
      self._write_tfdv(
          tfdv_path, train_dataset_name, train_features, eval_dataset_name,
          eval_features
      )
      tfma_src = tf_sources.TfmaSource(
          eval_result_paths=[tfma_path], metrics_exclude=['average_loss']
      )
      tfdv_src = tf_sources.TfdvSource(
          dataset_statistics_paths=[tfdv_path],
          features_include=['feature_name1', 'feature_name3']
      )
      model_src = tf_sources.ModelSource(pushed_model_path=pushed_model_path)

    mc = core.ModelCardToolkit(
        source=tf_sources.Source(
            tfma=tfma_src, tfdv=tfdv_src, model=model_src
        )
    ).scaffold_assets()  # yapf: disable

    with self.subTest(name='quantitative_analysis'):
      list_to_proto = lambda lst: [x.to_proto() for x in lst]
      expected_performance_metrics = [
          model_card.PerformanceMetric(
              type='post_export_metrics/example_count', value='2.0'
          )
      ]
      self.assertCountEqual(
          list_to_proto(mc.quantitative_analysis.performance_metrics),
          list_to_proto(expected_performance_metrics)
      )
      self.assertLen(mc.quantitative_analysis.graphics.collection, 1)

    with self.subTest(name='model_parameters.data'):
      self.assertLen(mc.model_parameters.data, 2)  # train and eval
      for dataset in mc.model_parameters.data:
        for graphic in dataset.graphics.collection:
          self.assertIsNotNone(
              graphic.image,
              msg=f'No image found for graphic: {dataset.name} {graphic.name}'
          )
          graphic.image = None  # ignore graphic.image for below assertions
      self.assertIn(
          model_card.Dataset(
              name=train_dataset_name, graphics=model_card.GraphicsCollection(
                  collection=[
                      model_card.Graphic(name='counts | feature_name1')
                  ]
              )
          ), mc.model_parameters.data
      )
      self.assertIn(
          model_card.Dataset(
              name=eval_dataset_name, graphics=model_card.GraphicsCollection(
                  collection=[
                      model_card.Graphic(name='counts | feature_name3')
                  ]
              )
          ), mc.model_parameters.data
      )
      self.assertNotIn(
          model_card.Dataset(
              name=eval_dataset_name, graphics=model_card.GraphicsCollection(
                  collection=[
                      model_card.Graphic(name='counts | feature_name2')
                  ]
              )
          ), mc.model_parameters.data
      )

    with self.subTest(name='model_details.path'):
      self.assertEqual(mc.model_details.path, pushed_model_path)

  def test_scaffold_assets_with_empty_source(self):
    core.ModelCardToolkit(source=tf_sources.Source()).scaffold_assets()

  def test_scaffold_assets_with_invalid_tfma_source(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Only one of TfmaSource.metrics_include and TfmaSource.metrics_exclude '
        'should be set.'
    ):
      core.ModelCardToolkit(
          source=tf_sources.Source(
              tfma=tf_sources.TfmaSource(
                  eval_result_paths=['dummy/path'], metrics_include=[
                      'false_positive_rate'
                  ], metrics_exclude=['false_negative_rate']
              )
          )
      )

  def test_scaffold_assets_with_invalid_tfdv_source(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Only one of TfdvSource.features_include and '
        'TfdvSource.features_exclude should be set.'
    ):
      core.ModelCardToolkit(
          source=tf_sources.Source(
              tfdv=tf_sources.TfdvSource(
                  dataset_statistics_paths=['dummy/path'], features_include=[
                      'brand_confidence'
                  ], features_exclude=['brand_prominence']
              )
          )
      )


if __name__ == '__main__':
  absltest.main()
else:
  # Manually pass and parse flags to prevent UnparsedFlagAccessError when using
  # pytest or unittest as a runner.
  flags.FLAGS(['--test_tmpdir'])
