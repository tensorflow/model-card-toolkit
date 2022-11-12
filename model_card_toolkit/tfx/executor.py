"""Model Card TFX Component Executor.

The ModelCard Executor handles the ModelCardToolkit workflow in the
ModelCardGenerator.
"""

from typing import Any, Dict, List, Optional

from model_card_toolkit import core
from model_card_toolkit.utils import source as src

from tfx import types
from tfx.dsl.components.base.base_executor import BaseExecutor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs

_DEFAULT_MODEL_CARD_FILE_NAME = 'model_card.html'


class Executor(BaseExecutor):
  """Executor for Model Card TFX component."""

  def _tfma_source(
      self, input_dict: Dict[str,
                             List[types.Artifact]]) -> Optional[src.TfmaSource]:
    """See base class."""
    if not input_dict.get(standard_component_specs.EVALUATION_KEY):
      return None
    else:
      return src.TfmaSource(model_evaluation_artifacts=input_dict[
          standard_component_specs.EVALUATION_KEY])

  def _tfdv_source(
      self, input_dict: Dict[str,
                             List[types.Artifact]]) -> Optional[src.TfdvSource]:
    """See base class."""
    if not input_dict.get(standard_component_specs.STATISTICS_KEY):
      return None
    else:
      return src.TfdvSource(example_statistics_artifacts=input_dict[
          standard_component_specs.STATISTICS_KEY])

  def _model_source(
      self,
      input_dict: Dict[str, List[types.Artifact]]) -> Optional[src.ModelSource]:
    """See base class."""
    if not input_dict.get(standard_component_specs.PUSHED_MODEL_KEY):
      return None
    else:
      return src.ModelSource(
          pushed_model_artifact=artifact_utils.get_single_instance(input_dict[
              standard_component_specs.PUSHED_MODEL_KEY]))

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Generate a model card for a TFX pipeline.

    This executes a Model Card Toolkit workflow, producing a `ModelCard`
    artifact.

    Args:
      input_dict: Input dict from key to a list of artifacts, including:
        - evaluation: TFMA output from an
          [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) component,
          used to populate quantitative analysis fields in the model card.
        - statistics: TFDV output from a
          [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen)
          component, used to populate dataset fields in the model card.
        - pushed_model: PushedModel output from a
          [Pusher](https://www.tensorflow.org/tfx/guide/pusher) component, used
          to populate model details in the the model card.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_card: An artifact referencing the directory containing the Model
          Card document, as well as the `ModelCard` used to construct the
          document.
      exec_properties: An optional dict of execution properties, including:
        - json: A JSON string containing `ModelCard` fields. This is
          particularly useful for fields that cannot be auto-populated from
          earlier TFX components. If a field is populated both by TFX and JSON,
          the JSON value will overwrite the TFX value. Use the [Model Card JSON
          schema](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/schema/v0.0.2/model_card.schema.json).
        - template_io: A list of input/output pairs. The input is the path to a
          [Jinja](https://jinja.palletsprojects.com/) template. Using data
          extracted from TFX components and `json`, this template is populated
          and saved as a model card. The output is a file name where the model
          card will be written to in the `model_card/` directory. By default,
          `ModelCardToolkit`'s default HTML template
          (`default_template.html.jinja`) and file name (`model_card.html`)
          are used.
    """

    # Initialize ModelCardToolkit
    mct = core.ModelCardToolkit(
        source=src.Source(
            tfma=self._tfma_source(input_dict),
            tfdv=self._tfdv_source(input_dict),
            model=self._model_source(input_dict)),
        output_dir=artifact_utils.get_single_instance(
            output_dict['model_card']).uri)
    template_io = exec_properties.get('template_io') or [
        (mct.default_template, _DEFAULT_MODEL_CARD_FILE_NAME)
    ]

    # Create model card assets from inputs
    mct.scaffold_assets(json=exec_properties.get('json'))
    for template_path, output_file in template_io:
      mct.export_format(
          template_path=template_path, output_file=output_file)
