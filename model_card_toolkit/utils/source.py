"""Sources to extract quantitative information for model cards.

These classes are inputs to the ModelCardToolkit, providing paths to data to
populate a ModelCard.
"""

import dataclasses
from typing import List, Optional, Text
from tfx.types import standard_artifacts

import ml_metadata as mlmd


@dataclasses.dataclass
class MlmdSource:
  """MLMD source to populate a model card.

  Attributes:
    store: A ml-metadata MetadataStore to retrieve metadata and lineage
      information about the model.
    model_uri: The path to the trained model used to generate the model card.
  """
  store: mlmd.MetadataStore
  model_uri: Text


@dataclasses.dataclass
class TfmaSource:
  """Source to extract TFMA eval result data for a model card.

  Provide exactly one of `eval_result_paths` or `model_evaluation_artifacts`.

  Attributes:
    eval_result_paths: The paths to the eval result output from TensorFlow Model
      Analysis or TFX Evaluator.
    file_format: Optional file extension to filter eval result files by.
    model_evaluation_artifacts: The MLMD artifacts from TensorFlow Model
      Analysis or TFX Evaluator.
    metrics_include: The list of metric names to include in the model card. By
      default, all metrics are included. Mutually exclusive with
      metrics_exclude.
    metrics_exclude: The list of metric names to exclude in the model card. By
      default, no metrics are excluded. Mutually exclusive with metrics_include.
  """
  eval_result_paths: List[Text] = dataclasses.field(default_factory=list)
  file_format: Optional[Text] = ''
  model_evaluation_artifacts: List[
      standard_artifacts.ModelEvaluation] = dataclasses.field(
          default_factory=list)
  metrics_include: List[Text] = dataclasses.field(default_factory=list)
  metrics_exclude: List[Text] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.eval_result_paths and not self.model_evaluation_artifacts:
      pass
    elif self.model_evaluation_artifacts and not self.eval_result_paths:
      self.eval_result_paths = [
          artifact.uri
          for artifact in self.model_evaluation_artifacts
      ]
    else:
      raise ValueError(
          'TfmaSource needs exactly one of eval_result_paths or '
          'model_evaluation_artifact'
      )

    if self.metrics_include and self.metrics_exclude:
      raise ValueError('Only one of TfmaSource.metrics_include and '
                       'TfmaSource.metrics_exclude should be set.')


@dataclasses.dataclass
class TfdvSource:
  """Sources to extract TFDV data for a model card.

  Provide exactly one of `dataset_statistics_paths` or
  `example_statistics_artifacts`.

  Attributes:
    dataset_statistics_paths: The paths to the output from TensorFlow Data
      Validation or TFX ExampleValidator.
    example_statistics_artifacts: The MLMD artifact from TensorFlow Data
      Validation or TFX ExampleValidator.
    features_include: The feature paths to include from the dataset statistics.
      By default, all features are included. Mutually exclusive with
      features_exclude.
    features_exclude: The feature paths to exclude from the dataset statistics.
      By default, all features are included. Mutually exclusive with
      features_include.
  """
  dataset_statistics_paths: List[Text] = dataclasses.field(default_factory=list)
  example_statistics_artifacts: List[
      standard_artifacts.ExampleStatistics] = dataclasses.field(
          default_factory=list)
  features_include: List[Text] = dataclasses.field(default_factory=list)
  features_exclude: List[Text] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    if self.dataset_statistics_paths and not self.example_statistics_artifacts:
      pass
    elif self.example_statistics_artifacts and not self.dataset_statistics_paths:
      self.dataset_statistics_paths = [
          artifact.uri
          for artifact in self.example_statistics_artifacts
      ]
    else:
      raise ValueError(
          'TfdvSource needs exactly one of dataset_statistics_paths or '
          'example_statistics_artifacts'
      )

    if self.features_include and self.features_exclude:
      raise ValueError('Only one of TfdvSource.features_include and '
                       'TfdvSource.features_exclude should be set.')


@dataclasses.dataclass
class ModelSource:
  """Sources to extract PushedModel data for a model card.

  Provide exactly one of `pushed_model_path` or `pushed_model_artifact`.

  Attributes:
    pushed_model_path: The path of a PushedModel.
    pushed_model_artifact: The MLMD artifact for a PushedModel.
  """
  pushed_model_path: Optional[Text] = ''
  pushed_model_artifact: Optional[standard_artifacts.PushedModel] = None

  def __post_init__(self):
    if self.pushed_model_path and not self.pushed_model_artifact:
      pass
    elif self.pushed_model_artifact and not self.pushed_model_path:
      self.pushed_model_path = self.pushed_model_artifact.uri
    else:
      raise ValueError(
          'ModelSource needs exactly one of pushed_model_path or '
          'pushed_model_artifact.'
      )


@dataclasses.dataclass
class Source:
  """Sources to extract data for a model card.

  Attributes:
    tfma: The source info for TFMA.
    tfdv: The source info for TFDV.
    model: The source info for PushedModel.
  """
  tfma: Optional[TfmaSource] = None
  tfdv: Optional[TfdvSource] = None
  model: Optional[ModelSource] = None
