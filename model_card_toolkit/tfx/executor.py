"""Model Card TFX Component Executor.

The ModelCard Executor is used to generate model cards in TFX pipelines.
"""

from typing import Any, Dict, List, Optional

from model_card_toolkit.model_card_toolkit import ModelCardToolkit
from model_card_toolkit.utils import source as src

from tfx import types
from tfx.dsl.components.base.base_executor import BaseExecutor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs


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
    artifact. This artifact references to a directory containing the Model Card
    document, as well as the `ModelCard` used to construct the document.

    Args:
      input_dict: Input dict from key to a list of artifacts, including:
        - evaluation: TFMA output, used to populate quantitative analysis fields
          in the model card.
        - statistics: TFDV output, used to populate dataset fields in the model
          card.
        - pushed_model: PushedModel output, used to populate model details in
          the model card.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_card: An artifact referencing the directory containing the Model
          Card document, as well as the `ModelCard` used to construct the
          document.
      exec_properties: An optional dict of execution properties, including:
        - json: A JSON object containing `ModelCard` fields. This is
          particularly useful for fields that cannot be auto-populated from
          earlier TFX components. If a field is populated both by TFX and JSON,
          the JSON value will overwrite the TFX value.
        - template_io: A list of input/output pairs. The input is a jinja
          template path to use when generating model card documents. The output
          is the file name to write the model card document to. If nothing is
          provided, `ModelCardToolkit`'s default HTML template and filename
          are used.
    """

    # Initialize ModelCardToolkit
    mct = ModelCardToolkit(
        source=src.Source(
            tfma=self._tfma_source(input_dict),
            tfdv=self._tfdv_source(input_dict),
            model=self._model_source(input_dict)),
        output_dir=artifact_utils.get_single_instance(
            output_dict['model_card']).uri)

    # Create model card assets from inputs
    mct.scaffold_assets(json=exec_properties.get('json'))
    for template_path, output_file in exec_properties.get(
        'template_io', [(None, None)]):
      mct.export_format(template_path=template_path, output_file=output_file)
