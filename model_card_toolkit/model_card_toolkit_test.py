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
"""Tests for model_card_toolkit."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from model_card_toolkit import model_card
from model_card_toolkit import model_card_toolkit
from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.tfx import artifact
from model_card_toolkit.utils import graphics
from model_card_toolkit.utils import source as src
from model_card_toolkit.utils.testdata import testdata_utils
from model_card_toolkit.utils.testdata.tfxtest import TfxTest
import tensorflow_model_analysis as tfma
from tfx.types import standard_artifacts

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class ModelCardToolkitTest(parameterized.TestCase, TfxTest):

  def setUp(self):
    super(ModelCardToolkitTest, self).setUp()
    test_dir = self.create_tempdir()
    self.tmp_db_path = os.path.join(test_dir, 'test_mlmd.db')
    self.mct_dir = test_dir.mkdir(os.path.join(test_dir,
                                               'model_card')).full_path

  def test_init_with_store_model_uri_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    unknown_model = 'unknown_model'
    with self.assertRaisesRegex(
        ValueError, f'"{unknown_model}" cannot be found in the `store`'):
      model_card_toolkit.ModelCardToolkit(
          mlmd_source=src.MlmdSource(store=store, model_uri=unknown_model))

  def test_scaffold_assets(self):
    output_dir = self.mct_dir
    mct = model_card_toolkit.ModelCardToolkit(output_dir=output_dir)
    self.assertEqual(mct.output_dir, output_dir)
    mct.scaffold_assets()
    self.assertIn('default_template.html.jinja',
                  os.listdir(os.path.join(output_dir, 'template/html')))
    self.assertIn('default_template.md.jinja',
                  os.listdir(os.path.join(output_dir, 'template/md')))
    self.assertIn('model_card.proto',
                  os.listdir(os.path.join(output_dir, 'data')))

  def test_scaffold_assets_with_json(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.mct_dir)
    mc = mct.scaffold_assets({'model_details': {'name': 'json_test',}})
    self.assertEqual(mc.model_details.name, 'json_test')

  @mock.patch.object(
      graphics, 'annotate_dataset_feature_statistics_plots', autospec=True)
  @mock.patch.object(graphics, 'annotate_eval_result_plots', autospec=True)
  def test_scaffold_assets_with_store(self, mock_annotate_data_stats,
                                      mock_annotate_eval_results):
    num_stat_artifacts = 2
    num_eval_artifacts = 1
    output_dir = self.mct_dir
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=output_dir,
        mlmd_source=src.MlmdSource(
            store=store, model_uri=testdata_utils.TFX_0_21_MODEL_URI))
    mc = mct.scaffold_assets()
    self.assertIsNotNone(mc.model_details.name)
    self.assertIsNotNone(mc.model_details.version.name)
    self.assertIn('default_template.html.jinja',
                  os.listdir(os.path.join(output_dir, 'template/html')))
    self.assertIn('default_template.md.jinja',
                  os.listdir(os.path.join(output_dir, 'template/md')))
    self.assertEqual(mock_annotate_data_stats.call_count, num_stat_artifacts)
    self.assertEqual(mock_annotate_eval_results.call_count, num_eval_artifacts)

  @parameterized.parameters(('', True), ('', False), ('tfrecord', True),
                            ('tfrecord', False))
  def test_scaffold_assets_with_source(self, output_file_format: str,
                                       artifacts: bool):
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
            num_buckets=2),
    ]

    if artifacts:
      connection_config = metadata_store_pb2.ConnectionConfig()
      connection_config.fake_database.SetInParent()
      mlmd_store = mlmd.MetadataStore(connection_config)
      self._write_tfma(tfma_path, output_file_format, add_metrics_callbacks,
                       mlmd_store)
      self._write_tfdv(tfdv_path, train_dataset_name, train_features,
                       eval_dataset_name, eval_features, mlmd_store)
      model_evaluation_artifacts = mlmd_store.get_artifacts_by_type(
          standard_artifacts.ModelEvaluation.TYPE_NAME)
      example_statistics_artifacts = mlmd_store.get_artifacts_by_type(
          standard_artifacts.ExampleStatistics.TYPE_NAME)
      pushed_model_artifact = standard_artifacts.PushedModel()
      pushed_model_artifact.uri = pushed_model_path
      tfma_src = src.TfmaSource(
          model_evaluation_artifacts=model_evaluation_artifacts,
          metrics_exclude=['average_loss'])
      tfdv_src = src.TfdvSource(
          example_statistics_artifacts=example_statistics_artifacts,
          features_include=['feature_name1', 'feature_name3'])
      model_src = src.ModelSource(pushed_model_artifact=pushed_model_artifact)
    else:
      self._write_tfma(tfma_path, output_file_format, add_metrics_callbacks)
      self._write_tfdv(tfdv_path, train_dataset_name, train_features,
                       eval_dataset_name, eval_features)
      tfma_src = src.TfmaSource(
          eval_result_paths=[tfma_path], metrics_exclude=['average_loss'])
      tfdv_src = src.TfdvSource(
          dataset_statistics_paths=[tfdv_path],
          features_include=['feature_name1', 'feature_name3'])
      model_src = src.ModelSource(pushed_model_path=pushed_model_path)

    mc = model_card_toolkit.ModelCardToolkit(
        source=src.Source(tfma=tfma_src, tfdv=tfdv_src,
                          model=model_src)).scaffold_assets()

    with self.subTest(name='quantitative_analysis'):
      list_to_proto = lambda lst: [x.to_proto() for x in lst]
      expected_performance_metrics = [
          model_card.PerformanceMetric(
              type='post_export_metrics/example_count', value='2.0')
      ]
      self.assertCountEqual(
          list_to_proto(mc.quantitative_analysis.performance_metrics),
          list_to_proto(expected_performance_metrics))
      self.assertLen(mc.quantitative_analysis.graphics.collection, 1)

    with self.subTest(name='model_parameters.data'):
      self.assertLen(mc.model_parameters.data, 2)  # train and eval
      for dataset in mc.model_parameters.data:
        for graphic in dataset.graphics.collection:
          self.assertIsNotNone(
              graphic.image,
              msg=f'No image found for graphic: {dataset.name} {graphic.name}')
          graphic.image = None  # ignore graphic.image for below assertions
      self.assertIn(
          model_card.Dataset(
              name=train_dataset_name,
              graphics=model_card.GraphicsCollection(collection=[
                  model_card.Graphic(name='counts | feature_name1')
              ])), mc.model_parameters.data)
      self.assertIn(
          model_card.Dataset(
              name=eval_dataset_name,
              graphics=model_card.GraphicsCollection(collection=[
                  model_card.Graphic(name='counts | feature_name3')
              ])), mc.model_parameters.data)
      self.assertNotIn(
          model_card.Dataset(
              name=eval_dataset_name,
              graphics=model_card.GraphicsCollection(collection=[
                  model_card.Graphic(name='counts | feature_name2')
              ])), mc.model_parameters.data)

    with self.subTest(name='model_details.path'):
      self.assertEqual(mc.model_details.path, pushed_model_path)

  def test_scaffold_assets_with_empty_source(self):
    model_card_toolkit.ModelCardToolkit(source=src.Source()).scaffold_assets()

  def test_scaffold_assets_with_invalid_tfma_source(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'Only one of TfmaSource.metrics_include and TfmaSource.metrics_exclude '
        'should be set.'):
      model_card_toolkit.ModelCardToolkit(
          source=src.Source(
              tfma=src.TfmaSource(
                  eval_result_paths=['dummy/path'],
                  metrics_include=['false_positive_rate'],
                  metrics_exclude=['false_negative_rate'])))

  def test_scaffold_assets_with_invalid_tfdv_source(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Only one of TfdvSource.features_include and '
        'TfdvSource.features_exclude should be set.'):
      model_card_toolkit.ModelCardToolkit(
          source=src.Source(
              tfdv=src.TfdvSource(
                  dataset_statistics_paths=['dummy/path'],
                  features_include=['brand_confidence'],
                  features_exclude=['brand_prominence'])))

  def test_update_model_card_with_valid_model_card(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.mct_dir)
    valid_model_card = mct.scaffold_assets()
    valid_model_card.model_details.name = 'My Model'
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')

    model_card_proto = model_card_pb2.ModelCard()
    with open(proto_path, 'rb') as f:
      model_card_proto.ParseFromString(f.read())
    self.assertEqual(model_card_proto, valid_model_card.to_proto())

  def test_update_model_card_with_valid_model_card_as_proto(self):
    valid_model_card = model_card_pb2.ModelCard()
    valid_model_card.model_details.name = 'My Model'

    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.mct_dir)
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')

    model_card_proto = model_card_pb2.ModelCard()
    with open(proto_path, 'rb') as f:
      model_card_proto.ParseFromString(f.read())
    self.assertEqual(model_card_proto, valid_model_card)

  def test_export_format(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=self.mct_dir,
        mlmd_source=src.MlmdSource(
            store=store, model_uri=testdata_utils.TFX_0_21_MODEL_URI))
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)
    result = mct.export_format()

    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')
    self.assertTrue(os.path.exists(proto_path))
    with open(proto_path, 'rb') as f:
      model_card_proto = model_card_pb2.ModelCard()
      model_card_proto.ParseFromString(f.read())
      self.assertEqual(model_card_proto.model_details.name, 'My Model')
    model_card_path = os.path.join(self.mct_dir, 'model_cards/model_card.html')
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_with_customized_template_and_output_name(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.mct_dir)
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)

    template_path = os.path.join(self.mct_dir,
                                 'template/html/default_template.html.jinja')
    output_file = 'my_model_card.html'
    result = mct.export_format(
        template_path=template_path, output_file=output_file)

    model_card_path = os.path.join(self.mct_dir, 'model_cards', output_file)
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_before_scaffold_assets(self):
    with self.assertRaises(ValueError):
      model_card_toolkit.ModelCardToolkit().export_format()

  def test_save_mlmd_without_mlmd(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    with self.assertRaises(ValueError):
      mct.save_mlmd(metadata_store_pb2.Artifact())

  def test_artifact_from_mlmd_source(self):

    # Create a MLMD store.
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mlmd_source = src.MlmdSource(
        store=store, model_uri=testdata_utils.TFX_0_21_MODEL_URI)

    # Instantiate the MCT instance.
    # We pass in an artifact_type_id, which should be ignored in favor of the
    # MLMD source.
    artifact_uri = self.tmpdir.full_path
    ignored_type_id = -1
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=artifact_uri,
        mlmd_source=mlmd_source,
        artifact_type_id=ignored_type_id)

    # Create ModelCard assets, which are stored as an Artifact.
    model_name = 'online_news_trainer.py'
    mc = mct.scaffold_assets(json={'model_details': {'name': model_name}})
    mct.export_format()

    mc_artifacts_from_store = store.get_artifacts_by_type('ModelCard')
    with self.subTest(name='check_artifact_saved_to_store'):
      self.assertLen(mc_artifacts_from_store, 1)
      self.assertEqual(mct.artifact, mc_artifacts_from_store[0])

    with self.subTest(name='check_artifact_fields_have_expected_values'):
      self.assertEqual(mct.artifact.uri, artifact_uri)
      self.assertStartsWith(mct.artifact.name, model_name)

    with self.subTest(name='check_type_id_was_ignored'):
      self.assertNotEqual(mct.artifact.type_id, ignored_type_id)

    with self.subTest(name='check_artifact_directory_contents'):
      self.assertContainsSubset(['data', 'model_cards'],
                                os.listdir(mct.artifact.uri))
      self.assertContainsSubset(['model_card.proto'],
                                os.listdir(
                                    os.path.join(mct.artifact.uri, 'data')))
      self.assertContainsSubset(['model_card.html'],
                                os.listdir(
                                    os.path.join(mct.artifact.uri,
                                                 'model_cards')))
      with self.subTest(name='check_model_card_proto'):
        model_card_from_mlmd = model_card_pb2.ModelCard()
        with open(
            os.path.join(mct.artifact.uri, 'data', 'model_card.proto'),
            'rb') as f:
          model_card_from_mlmd.ParseFromString(f.read())
        self.assertEqual(mc.to_proto(), model_card_from_mlmd)

    with self.subTest(name='check_save_mlmd_returns_same_artifact'):
      self.assertEqual(mct.save_mlmd(mct.artifact), mct.artifact)

  def test_artifact_from_type_id(self):

    # Create a MLMD store with ModelCard artifact type.
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    type_id = artifact.put_model_card_artifact_type(store)

    # Instantiate the MCT instance, using artifact type instead of MLMD store.
    artifact_uri = self.tmpdir.full_path
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=artifact_uri, artifact_type_id=type_id)

    # Create ModelCard assets, which are stored as an Artifact.
    model_name = 'online_news_trainer.py'
    mc = mct.scaffold_assets(json={'model_details': {'name': model_name}})
    mct.export_format()

    with self.subTest(name='check_artifact_fields_have_expected_values'):
      self.assertEqual(mct.artifact.uri, artifact_uri)
      self.assertStartsWith(mct.artifact.name, model_name)
      self.assertEqual(mct.artifact.type_id, type_id)

    with self.subTest(name='check_artifact_directory_contents'):
      self.assertContainsSubset(['data', 'model_cards'],
                                os.listdir(mct.artifact.uri))
      self.assertContainsSubset(['model_card.proto'],
                                os.listdir(
                                    os.path.join(mct.artifact.uri, 'data')))
      self.assertContainsSubset(['model_card.html'],
                                os.listdir(
                                    os.path.join(mct.artifact.uri,
                                                 'model_cards')))
      with self.subTest(name='check_model_card_proto'):
        model_card_from_mlmd = model_card_pb2.ModelCard()
        with open(
            os.path.join(mct.artifact.uri, 'data', 'model_card.proto'),
            'rb') as f:
          model_card_from_mlmd.ParseFromString(f.read())
        self.assertEqual(mc.to_proto(), model_card_from_mlmd)

    with self.subTest(name='check_save_mlmd_fails_without_mlmd_store'):
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'Cannot save to MLMD store because MLMD store was not '
          'registered to ModelCardToolkit instance.'):
        mct.save_mlmd(mct.artifact)


if __name__ == '__main__':
  absltest.main()
