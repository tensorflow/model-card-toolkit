"""Tests for model_card_toolkit.tfx.executor."""

import os
from absl.testing import absltest
from absl.testing import parameterized
from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.tfx import artifact as artifact_utils
from model_card_toolkit.tfx import executor
from model_card_toolkit.utils.testdata.tfxtest import TfxTest
import tensorflow_model_analysis as tfma
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ExecutorTest(parameterized.TestCase, TfxTest):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self.mct_executor = executor.Executor()
    mlmd_store = self._set_up_mlmd()

    # Write template
    self.template_file = self.create_tempdir().create_file()
    self.template_file.write_text('hello world')

    # Write TFMA metrics to store
    tfma_path = os.path.join(self.tmpdir, 'tfma')
    add_metrics_callbacks = [
        tfma.post_export_metrics.example_count(),
        tfma.post_export_metrics.calibration_plot_and_prediction_histogram(
            num_buckets=2)
    ]
    self._write_tfma(tfma_path, '', add_metrics_callbacks, mlmd_store)

    # Write TFDV statistics to store
    tfdv_path = os.path.join(self.tmpdir, 'tfdv')
    self.train_dataset_name = 'Dataset-Split-train'
    self.train_features = ['feature_name1']
    self.eval_dataset_name = 'Dataset-Split-eval'
    self.eval_features = ['feature_name2']
    self._write_tfdv(tfdv_path, self.train_dataset_name, self.train_features,
                     self.eval_dataset_name, self.eval_features, mlmd_store)

    self.eval_artifacts = mlmd_store.get_artifacts_by_type(
        standard_artifacts.ModelEvaluation.TYPE_NAME)
    self.example_stats_artifacts = mlmd_store.get_artifacts_by_type(
        standard_artifacts.ExampleStatistics.TYPE_NAME)

    self.pushed_model_path = os.path.join(self.tmpdir, 'pushed_model')
    self.pushed_model_artifact = standard_artifacts.PushedModel()
    self.pushed_model_artifact.uri = self.pushed_model_path

    self.model_card_artifact = artifact_utils.create_and_save_artifact(
        artifact_name=self.pushed_model_artifact.name + '_model_card',
        artifact_uri=self.create_tempdir().full_path,
        store=mlmd_store)

  @parameterized.named_parameters(
      dict(
          testcase_name='fullInput',
          eval_artifacts=True,
          example_stats_artifacts=True,
          pushed_model_artifact=True,
          exec_props=True),
      dict(
          testcase_name='emptyInput',
          eval_artifacts=False,
          example_stats_artifacts=False,
          pushed_model_artifact=False,
          exec_props=False),
      dict(
          testcase_name='partialInput',
          eval_artifacts=False,
          example_stats_artifacts=True,
          pushed_model_artifact=False,
          exec_props=True))
  def test_do(self, eval_artifacts: bool, example_stats_artifacts: bool,
              pushed_model_artifact: bool, exec_props: bool):

    input_dict = {}
    if eval_artifacts:
      input_dict[standard_component_specs.EVALUATION_KEY] = self.eval_artifacts
    if example_stats_artifacts:
      input_dict[standard_component_specs
                 .STATISTICS_KEY] = self.example_stats_artifacts
    if pushed_model_artifact:
      input_dict[standard_component_specs.PUSHED_MODEL_KEY] = [
          self.pushed_model_artifact
      ]

    output_dict = {'model_card': [self.model_card_artifact]}

    exec_properties = {}
    if exec_props:
      exec_properties['json'] = {'model_details': {'name': 'json_test',}}
      exec_properties['template_io'] = [(self.template_file.full_path,
                                         'my_cool_model_card.html')]

    # Call MCT Executor
    self.mct_executor.Do(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties)

    # Verify model card proto and document were generated
    self.assertIn(
        'model_card.proto',
        os.listdir(os.path.join(self.model_card_artifact.uri, 'data')))
    self.assertIn(
        'default_template.html.jinja',
        os.listdir(
            os.path.join(self.model_card_artifact.uri, 'template', 'html')))

    model_card_proto = model_card_pb2.ModelCard()
    with open(
        os.path.join(self.model_card_artifact.uri, 'data', 'model_card.proto'),
        'rb') as f:
      model_card_proto.ParseFromString(f.read())

    with self.subTest(name='exec_props'):
      model_card_dir = os.path.join(self.model_card_artifact.uri, 'model_cards')
      if exec_props:
        self.assertEqual(model_card_proto.model_details.name, 'json_test')
        model_card_file_name = 'my_cool_model_card.html'
      else:
        model_card_file_name = 'model_card.html'
      self.assertIn(model_card_file_name, os.listdir(model_card_dir))
      model_card_filepath = os.path.join(model_card_dir,
                                         model_card_file_name)
      with open(model_card_filepath) as f:
        model_card_content = f.read()
      if exec_props:
        self.assertEqual(model_card_content, 'hello world')
      else:
        self.assertStartsWith(model_card_content, '<!DOCTYPE html>')

    if eval_artifacts:
      with self.subTest(name='eval_artifacts'):
        self.assertCountEqual(
            model_card_proto.quantitative_analysis.performance_metrics, [
                model_card_pb2.PerformanceMetric(
                    type='post_export_metrics/example_count',
                    value='2.0',
                    confidence_interval=model_card_pb2.ConfidenceInterval()),
                model_card_pb2.PerformanceMetric(
                    type='average_loss',
                    value='0.5',
                    confidence_interval=model_card_pb2.ConfidenceInterval())
            ])
        self.assertLen(
            model_card_proto.quantitative_analysis.graphics.collection, 2)

    if example_stats_artifacts:
      with self.subTest(name='example_stats_artifacts.data'):
        self.assertLen(model_card_proto.model_parameters.data,
                       2)  # train and eval
        for dataset in model_card_proto.model_parameters.data:
          for graphic in dataset.graphics.collection:
            self.assertIsNotNone(
                graphic.image,
                msg=f'No image found for graphic: {dataset.name} {graphic.name}'
            )
            graphic.image = bytes()  # ignore graphic.image for below assertions
        self.assertIn(
            model_card_pb2.Dataset(
                name=self.train_dataset_name,
                graphics=model_card_pb2.GraphicsCollection(collection=[
                    model_card_pb2.Graphic(
                        name='counts | feature_name1', image='')
                ]),
                sensitive=model_card_pb2.SensitiveData()),
            model_card_proto.model_parameters.data)
        self.assertIn(
            model_card_pb2.Dataset(
                name=self.eval_dataset_name,
                graphics=model_card_pb2.GraphicsCollection(collection=[
                    model_card_pb2.Graphic(
                        name='counts | feature_name2', image='')
                ]),
                sensitive=model_card_pb2.SensitiveData()),
            model_card_proto.model_parameters.data)

    if pushed_model_artifact:
      with self.subTest(name='pushed_model_artifact'):
        self.assertEqual(model_card_proto.model_details.path,
                         self.pushed_model_path)


if __name__ == '__main__':
  absltest.main()
