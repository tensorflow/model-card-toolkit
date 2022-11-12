"""Model Card TFX Component.

The ModelCardGenerator is used to generate model cards in a TFX pipeline.
"""

from typing import Any, List, Tuple, Optional

from model_card_toolkit.tfx import artifact
from model_card_toolkit.tfx import executor

from tfx import types
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs

MODEL_CARD_KEY = 'model_card'


class ModelCardGeneratorSpec(component_spec.ComponentSpec):
  """Component spec for the ModelCardGenerator."""
  PARAMETERS = {
      'json':
          component_spec.ExecutionParameter(type=str, optional=True),
      # template_io's type is List[Tuple[str, str]],
      # but we need List[Any] to pass ExecutionParameter.type_check().
      # See below link for details.
      # https://github.com/tensorflow/tfx/blob/4ff5e97b09540ff8a858076a163ecdf209716324/tfx/types/component_spec.py#L308
      'template_io':
          component_spec.ExecutionParameter(
              type=List[Any], optional=True)
  }
  INPUTS = {
      standard_component_specs.STATISTICS_KEY:
          component_spec.ChannelParameter(
              type=standard_artifacts.ExampleStatistics, optional=True),
      standard_component_specs.EVALUATION_KEY:
          component_spec.ChannelParameter(
              type=standard_artifacts.ModelEvaluation, optional=True),
      standard_component_specs.PUSHED_MODEL_KEY:
          component_spec.ChannelParameter(
              type=standard_artifacts.PushedModel, optional=True),
  }
  OUTPUTS = {
      MODEL_CARD_KEY:
          component_spec.ChannelParameter(type=artifact.ModelCard),
  }


class ModelCardGenerator(BaseComponent):
  """A TFX component to generate a model card.

  The `ModelCardGenerator` is a [TFX
  Component](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines#component)
  that generates model cards.

  The model cards are written to a `ModelCard` artifact that can be fetched
  from the `outputs['model_card]'` property.

  Example:

  ```py
  context = InteractiveContext()
  ...
  mc_gen = ModelCardGenerator(
      statistics=statistics_gen.outputs['statistics'],
      evaluation=evaluator.outputs['evaluation'],
      pushed_model=pusher.outputs['pushed_model'],
      json="{'model_details': {'name': 'my_model'}}",
      template_io=[
          ('html/default_template.html.jinja', 'model_card.html'),
          ('md/default_template.md.jinja', 'model_card.md')
      ]
      )
  context.run(mc_gen)
  mc_artifact = mc_gen.outputs['model_card'].get()[0]
  mc_path = os.path.join(mc_artifact.uri, 'model_card', 'model_card.html')
  with open(mc_path) as f:
    mc_content = f.readlines()
  ```
  """

  SPEC_CLASS = ModelCardGeneratorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               evaluation: Optional[types.Channel] = None,
               statistics: Optional[types.Channel] = None,
               pushed_model: Optional[types.Channel] = None,
               json: Optional[str] = None,
               template_io: Optional[List[Tuple[str, str]]] = None
              ):
    """Generate a model card for a TFX pipeline.

    This executes a Model Card Toolkit workflow, producing a `ModelCard`
    artifact.

    Model card generation is partially automated from TFX, using the
    `ExampleStatistics`, `ModelEvaluation`, and `PushedModel` artifacts. Model
    card fields may be manually populated using the `json` arg. See the Args
    section for more details.

    To use custom model card templates, use the `template_io` arg.
    `ModelCardGenerator` can generate multiple model cards per execution.

    Args:
      evaluation: TFMA output from an
        [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) component,
        used to populate quantitative analysis fields in the model card.
      statistics: TFDV output from a
        [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen)
        component, used to populate dataset fields in the model card.
      pushed_model: PushedModel output from a
        [Pusher](https://www.tensorflow.org/tfx/guide/pusher) component, used to
        populate model details in the the model card.
      json: A JSON string containing `ModelCard` fields. This is particularly
        useful for fields that cannot be auto-populated from earlier TFX
        components. If a field is populated both by TFX and JSON, the JSON value
        will overwrite the TFX value. Use the [Model Card JSON
        schema](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/schema/v0.0.2/model_card.schema.json).
      template_io: A list of input/output pairs. The input is the path to a
        [Jinja](https://jinja.palletsprojects.com/) template. Using data
        extracted from TFX components and `json`, this template is populated and
        saved as a model card. The output is a file name where the model card
        will be written to in the `model_card/` directory. By default,
        `ModelCardToolkit`'s default HTML template
        (`default_template.html.jinja`) and file name (`model_card.html`) are
        used.
    """
    spec = ModelCardGeneratorSpec(
        evaluation=evaluation,
        statistics=statistics,
        pushed_model=pushed_model,
        model_card=types.Channel(type=artifact.ModelCard),
        json=json,
        template_io=template_io)
    super(ModelCardGenerator, self).__init__(spec=spec)
