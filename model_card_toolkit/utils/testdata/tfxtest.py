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
"""A helper class for testing interop with TFX pipelines."""

import os
from typing import Any, Callable, List, Optional

import apache_beam as beam
from model_card_toolkit.utils.tfx_util import _TFX_METRICS_TYPE
from model_card_toolkit.utils.tfx_util import _TFX_STATS_TYPE
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tfx_bsl.tfxio import raw_tf_record

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


class TfxTest(tfma.eval_saved_model.testutil.TensorflowModelAnalysisTest):
  """A helper class for testing interop with TFX pipelines."""

  def setUp(self):
    super(TfxTest, self).setUp()
    self.tmp_db_path = os.path.join(self.create_tempdir(), 'test_mlmd.db')
    self.tmpdir = self.create_tempdir()

  def _set_up_mlmd(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    return mlmd.MetadataStore(connection_config)

  def _put_artifact(self, store: mlmd.MetadataStore, type_name: str,
                    uri: str) -> None:
    type_id = store.put_artifact_type(
        metadata_store_pb2.ArtifactType(name=type_name))
    store.put_artifacts(
        [metadata_store_pb2.Artifact(uri=uri, type_id=type_id)])

  def _write_tfma(self,
                  tfma_path: str,
                  output_file_format: str,
                  add_metrics_callbacks: List[Callable[..., Any]],
                  store: Optional[mlmd.MetadataStore] = None) -> None:
    """Runs a sample TFMA job and stores output.

    This uses a trivial inputs (two examples, with prediction/label = 0/1 and
    1/1). and writes metrics and plots to the specified path.

    Args:
      tfma_path: The path to save the TFMA output to.
      output_file_format: The format to save TFMA output to. See [TFMA API
        Docs](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/writers/MetricsPlotsAndValidationsWriter)
          for the most up-to-date reference. If the empty string, 'tfrecord'
          will be used.
      add_metrics_callbacks: TFMA metric callbacks to compute. See [TFMA API
        Docs](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/post_export_metrics)
          for examples.
      store: The MLMD store to save the TFMA output artifact.
    """
    if not output_file_format:
      output_file_format = 'tfrecord'
    _, eval_saved_model_path = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            export_path=None,
            eval_export_path=os.path.join(self.tmpdir, 'eval_export_dir')))
    eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=eval_saved_model_path,
        add_metrics_callbacks=add_metrics_callbacks)
    extractors = [
        tfma.extractors.legacy_predict_extractor.PredictExtractor(
            eval_shared_model, eval_config=eval_config),
        tfma.extractors.unbatch_extractor.UnbatchExtractor(),
        tfma.extractors.slice_key_extractor.SliceKeyExtractor(),
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
            add_metrics_callbacks=eval_shared_model.add_metrics_callbacks),
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
      self._put_artifact(store, _TFX_METRICS_TYPE, tfma_path)

  def _write_tfdv(self,
                  tfdv_path: str,
                  train_dataset_name: str,
                  train_features: List[str],
                  eval_dataset_name: str,
                  eval_features: List[str],
                  store: Optional[mlmd.MetadataStore] = None) -> None:
    """Runs a sample TFDV job and stores output.

    For the training and evaluation datasets, for each feature, this creates a
    trivial TFDV histogram with three buckets. It writes this output to the
    specified path.

    Args:
      tfdv_path: The path to save the TFDV output to.
      train_dataset_name: The name to give the training dataset in the TFDV
        analysis.
      train_features: The names of the features in the training dataset.
      eval_dataset_name: The name to give the evaluation dataset in the TFDV
        analysis.
      eval_features: The names of the features in the evaluation dataset.
      store: The MLMD store to save the TFDV output artifact.
    """

    def _write(dataset_name: str, features: List[str], split_name: str):
      stats = statistics_pb2.DatasetFeatureStatistics()
      stats.name = dataset_name
      for feature in features:
        stat_feature = stats.features.add()
        stat_feature.name = feature
        stat_feature.string_stats.rank_histogram.buckets.extend([
            statistics_pb2.RankHistogram.Bucket(
                low_rank=0, high_rank=0, label='a', sample_count=4.0),
            statistics_pb2.RankHistogram.Bucket(
                low_rank=1, high_rank=1, label='b', sample_count=3.0),
            statistics_pb2.RankHistogram.Bucket(
                low_rank=2, high_rank=2, label='c', sample_count=2.0)
        ])
      stats_list = statistics_pb2.DatasetFeatureStatisticsList(datasets=[stats])
      stats_file = os.path.join(tfdv_path, split_name, 'FeatureStats.pb')
      os.makedirs(os.path.dirname(stats_file), exist_ok=True)
      with open(stats_file, mode='wb') as f:
        f.write(stats_list.SerializeToString())

    _write(train_dataset_name, train_features, 'Split-train')
    _write(eval_dataset_name, eval_features, 'Split-eval')

    if store:
      self._put_artifact(store, _TFX_STATS_TYPE, tfdv_path)
