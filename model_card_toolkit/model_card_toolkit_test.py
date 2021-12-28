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
from typing import List, Optional, Text
from unittest import mock
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam

from model_card_toolkit import model_card
from model_card_toolkit import model_card_toolkit
from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.utils import graphics
from model_card_toolkit.utils import source as src
from model_card_toolkit.utils.testdata import testdata_utils

import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tfx.types import standard_artifacts
from tfx_bsl.tfxio import raw_tf_record

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class ModelCardToolkitTest(
    parameterized.TestCase,
    tfma.eval_saved_model.testutil.TensorflowModelAnalysisTest):

  def setUp(self):
    super(ModelCardToolkitTest, self).setUp()
    self.tmp_db_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    f'test_mlmd_{uuid.uuid4()}.db')
    self.tmpdir = os.path.join(absltest.get_default_test_tmpdir(),
                               f'model_card_{uuid.uuid4()}')
    if not os.path.exists(self.tmpdir):
      os.makedirs(self.tmpdir)

  def _write_tfma(self,
                  tfma_path: Text,
                  output_file_format: Text,
                  store: Optional[mlmd.MetadataStore] = None):
    _, eval_saved_model_path = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            export_path=None,
            eval_export_path=os.path.join(self.tmpdir, 'eval_export_dir')))
    eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_saved_model_path,
        add_metrics_callbacks=[
            tfma.post_export_metrics.example_count(),
            tfma.post_export_metrics.calibration_plot_and_prediction_histogram(
                num_buckets=2)
        ])
    extractors = [
        tfma.extractors.legacy_predict_extractor.PredictExtractor(
            eval_shared_model, eval_config=eval_config),
        tfma.extractors.unbatch_extractor.UnbatchExtractor(),
        tfma.extractors.slice_key_extractor.SliceKeyExtractor()
    ]
    evaluators = [
        tfma.evaluators.legacy_metrics_and_plots_evaluator
        .MetricsAndPlotsEvaluator(eval_shared_model)
    ]
    writers = [
        tfma.writers.MetricsPlotsAndValidationsWriter(
            output_paths={
                'metrics': os.path.join(tfma_path, 'metrics'),
                'plots': os.path.join(tfma_path, 'plots')
            },
            output_file_format=output_file_format,
            eval_config=eval_config,
            add_metrics_callbacks=eval_shared_model.add_metrics_callbacks)
    ]

    tfx_io = raw_tf_record.RawBeamRecordTFXIO(
        physical_format='inmemory',
        raw_record_column_name='__raw_record__',
        telemetry_descriptors=['TFMATest'])
    with beam.Pipeline() as pipeline:
      example1 = self._makeExample(prediction=0.0, label=1.0)
      example2 = self._makeExample(prediction=1.0, label=1.0)
      _ = (
          pipeline
          | 'Create' >> beam.Create([
              example1.SerializeToString(),
              example2.SerializeToString(),
          ])
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'ExtractEvaluateAndWriteResults' >>
          tfma.ExtractEvaluateAndWriteResults(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              extractors=extractors,
              evaluators=evaluators,
              writers=writers))

    if store:
      eval_type = metadata_store_pb2.ArtifactType()
      eval_type.name = standard_artifacts.ModelEvaluation.TYPE_NAME
      eval_type_id = store.put_artifact_type(eval_type)

      artifact = metadata_store_pb2.Artifact()
      artifact.uri = tfma_path
      artifact.type_id = eval_type_id
      store.put_artifacts([artifact])

  def _write_tfdv(self,
                  tfdv_path: Text,
                  train_dataset_name: Text,
                  train_features: List[Text],
                  eval_dataset_name: Text,
                  eval_features: List[Text],
                  store: Optional[mlmd.MetadataStore] = None):

    a_bucket = statistics_pb2.RankHistogram.Bucket(
        low_rank=0, high_rank=0, label='a', sample_count=4.0)
    b_bucket = statistics_pb2.RankHistogram.Bucket(
        low_rank=1, high_rank=1, label='b', sample_count=3.0)
    c_bucket = statistics_pb2.RankHistogram.Bucket(
        low_rank=2, high_rank=2, label='c', sample_count=2.0)

    train_stats = statistics_pb2.DatasetFeatureStatistics()
    train_stats.name = train_dataset_name
    for feature in train_features:
      train_stats.features.add()
      train_stats.features[0].name = feature
      train_stats.features[0].string_stats.rank_histogram.buckets.extend(
          [a_bucket, b_bucket, c_bucket])
    train_stats_list = statistics_pb2.DatasetFeatureStatisticsList(
        datasets=[train_stats])
    train_stats_file = os.path.join(tfdv_path, 'Split-train', 'FeatureStats.pb')
    os.makedirs(os.path.dirname(train_stats_file), exist_ok=True)
    with open(train_stats_file, mode='wb') as f:
      f.write(train_stats_list.SerializeToString())

    eval_stats = statistics_pb2.DatasetFeatureStatistics()
    eval_stats.name = eval_dataset_name
    for feature in eval_features:
      eval_stats.features.add()
      eval_stats.features[0].path.step.append(feature)
      eval_stats.features[0].string_stats.rank_histogram.buckets.extend(
          [a_bucket, b_bucket, c_bucket])
    eval_stats_list = statistics_pb2.DatasetFeatureStatisticsList(
        datasets=[eval_stats])
    eval_stats_file = os.path.join(tfdv_path, 'Split-eval', 'FeatureStats.pb')
    os.makedirs(os.path.dirname(eval_stats_file), exist_ok=True)
    with open(eval_stats_file, mode='wb') as f:
      f.write(eval_stats_list.SerializeToString())

    if store:
      stats_type = metadata_store_pb2.ArtifactType()
      stats_type.name = standard_artifacts.ExampleStatistics.TYPE_NAME
      stats_type_id = store.put_artifact_type(stats_type)

      artifact = metadata_store_pb2.Artifact()
      artifact.uri = tfdv_path
      artifact.type_id = stats_type_id
      store.put_artifacts([artifact])

  def test_init_with_store_model_uri_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    unknown_model = 'unknown_model'
    with self.assertRaisesRegex(
        ValueError, f'"{unknown_model}" cannot be found in the `store`'):
      model_card_toolkit.ModelCardToolkit(
          mlmd_source=src.MlmdSource(store=store, model_uri=unknown_model))

  def test_scaffold_assets(self):
    output_dir = self.tmpdir
    mct = model_card_toolkit.ModelCardToolkit(output_dir=output_dir)
    self.assertEqual(mct.output_dir, output_dir)
    mc = mct.scaffold_assets()  # pylint: disable=unused-variable
    self.assertIn('default_template.html.jinja',
                  os.listdir(os.path.join(output_dir, 'template/html')))
    self.assertIn('default_template.md.jinja',
                  os.listdir(os.path.join(output_dir, 'template/md')))
    self.assertIn('model_card.proto',
                  os.listdir(os.path.join(output_dir, 'data')))

  @mock.patch.object(
      graphics, 'annotate_dataset_feature_statistics_plots', autospec=True)
  @mock.patch.object(graphics, 'annotate_eval_result_plots', autospec=True)
  def test_scaffold_assets_with_store(self, mock_annotate_data_stats,
                                      mock_annotate_eval_results):
    num_stat_artifacts = 2
    num_eval_artifacts = 1
    output_dir = self.tmpdir
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
  def test_scaffold_assets_with_source(self, output_file_format: Text,
                                       artifacts: bool):
    if artifacts:
      connection_config = metadata_store_pb2.ConnectionConfig()
      connection_config.fake_database.SetInParent()
      mlmd_store = mlmd.MetadataStore(connection_config)
    else:
      mlmd_store = None

    train_dataset_name = 'Dataset-Split-train'
    train_features = ['feature_name1']
    eval_dataset_name = 'Dataset-Split-eval'
    eval_features = ['feature_name2']

    tfma_path = os.path.join(self.tmpdir, 'tfma')
    tfdv_path = os.path.join(self.tmpdir, 'tfdv')
    pushed_model_path = os.path.join(self.tmpdir, 'pushed_model')
    self._write_tfma(tfma_path, output_file_format, mlmd_store)
    self._write_tfdv(tfdv_path, train_dataset_name, train_features,
                     eval_dataset_name, eval_features, mlmd_store)

    if artifacts:
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
          features_include=['feature_name1'])
      model_src = src.ModelSource(pushed_model_artifact=pushed_model_artifact)
    else:
      tfma_src = src.TfmaSource(
          eval_result_paths=[tfma_path], metrics_exclude=['average_loss'])
      tfdv_src = src.TfdvSource(
          dataset_statistics_paths=[tfdv_path],
          features_include=['feature_name1'])
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
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    valid_model_card = mct.scaffold_assets()
    valid_model_card.model_details.name = 'My Model'
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.tmpdir, 'data/model_card.proto')

    model_card_proto = model_card_pb2.ModelCard()
    with open(proto_path, 'rb') as f:
      model_card_proto.ParseFromString(f.read())
    self.assertEqual(model_card_proto, valid_model_card.to_proto())

  def test_update_model_card_with_valid_model_card_as_proto(self):
    valid_model_card = model_card_pb2.ModelCard()
    valid_model_card.model_details.name = 'My Model'

    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.tmpdir, 'data/model_card.proto')

    model_card_proto = model_card_pb2.ModelCard()
    with open(proto_path, 'rb') as f:
      model_card_proto.ParseFromString(f.read())
    self.assertEqual(model_card_proto, valid_model_card)

  def test_export_format(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=self.tmpdir,
        mlmd_source=src.MlmdSource(
            store=store, model_uri=testdata_utils.TFX_0_21_MODEL_URI))
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)
    result = mct.export_format()

    proto_path = os.path.join(self.tmpdir, 'data/model_card.proto')
    self.assertTrue(os.path.exists(proto_path))
    with open(proto_path, 'rb') as f:
      model_card_proto = model_card_pb2.ModelCard()
      model_card_proto.ParseFromString(f.read())
      self.assertEqual(model_card_proto.model_details.name, 'My Model')
    model_card_path = os.path.join(self.tmpdir, 'model_cards/model_card.html')
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_with_customized_template_and_output_name(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)

    template_path = os.path.join(self.tmpdir,
                                 'template/html/default_template.html.jinja')
    output_file = 'my_model_card.html'
    result = mct.export_format(
        template_path=template_path, output_file=output_file)

    model_card_path = os.path.join(self.tmpdir, 'model_cards', output_file)
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_before_scaffold_assets(self):
    with self.assertRaises(ValueError):
      model_card_toolkit.ModelCardToolkit().export_format()


if __name__ == '__main__':
  absltest.main()
