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
"""Model Cards Toolkit.

The model cards toolkit (MCT) provides a set of utilities to help users
generate Model Cards from trained models within ML pipelines.
"""

import json
import os
import pkgutil
import tempfile
from typing import Optional, Text

from absl import logging
import jinja2

from model_card_toolkit.model_card import ModelCard
from model_card_toolkit.utils import graphics
from model_card_toolkit.utils import tfx_util
from model_card_toolkit.utils import validation

import ml_metadata as mlmd

# Constants about provided UI templates.
_UI_TEMPLATES = (
    'template/html/default_template.html.jinja',
    'template/md/default_template.md.jinja',
)
_DEFAULT_UI_TEMPLATE_FILE = os.path.join('html', 'default_template.html.jinja')
# Constants about Model Cards Toolkit Assets (MCTA).
_MCTA_JSON_FILE = 'data/model_card.json'
_MCTA_TEMPLATE_DIR = 'template'
_MCTA_RESOURCE_DIR = 'resources/plots'
# Constants about the final generated model cards.
_DEFAULT_MODEL_CARD_FILE_NAME = 'model_card.html'
_MODEL_CARDS_DIR = 'model_cards/'


class ModelCardToolkit():
  """ModelCardToolkit provides utilities to generate a ModelCard.

  ModelCardToolkit is a tool for ML practitioners to create Model Cards,
  documentation for model information such as owners, use cases, training and
  evaluation data, performance, etc. A Model Card document can be displayed in
  output formats including HTML, Markdown, etc.

  The ModelCardToolkit includes an API designed for a human-in-the-loop process
  to elaborate the ModelCard. If model training is integrated with ML Metadata
  (e.g., TFX pipelines), the ModelCardToolkit can further populate ModelCard
  fields by extract metadata and lineage from the model's MLMD instance.

  The ModelCardToolkit organizes the ModelCard assets (e.g., structured data,
  plots, and UI templates) in a user-specified directory, and updates them
  incrementally via its API.


  Example usage:

  ```python
  import model_card_toolkit

  # Initialize the Model Card Toolkit with a path to store generate assets
  model_card_output_path = ...
  mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

  # Initialize the ModelCard, which can be freely populated
  model_card = mct.scaffold_assets()
  model_card.model_details.name = 'My Model'

  # Write the model card data to a JSON file
  mct.update_model_card_json(model_card)

  # Return the model card document as an HTML page
  html = mct.export_format()
  ```
  """

  def __init__(self,
               output_dir: Optional[Text] = None,
               mlmd_store: Optional[mlmd.MetadataStore] = None,
               model_uri: Optional[Text] = None):
    """Initializes the ModelCardToolkit.

    Args:
      output_dir: The MCT assets path where the json file and templates are
        written to. If not given, a temp directory is used.
      mlmd_store: A ml-metadata MetadataStore to retrieve metadata and lineage
        information about the model stored at `model_uri`. If given, a set of
        model card properties can be auto-populated from the `mlmd_store`.
      model_uri: The path to the trained model to generate model cards.

    Raises:
      ValueError: If `mlmd_store` is given and the `model_uri` cannot be
        resolved as a model artifact in the metadata store.
    """
    self.output_dir = output_dir or tempfile.mkdtemp()
    self._mcta_json_file = os.path.join(self.output_dir, _MCTA_JSON_FILE)
    self._mcta_template_dir = os.path.join(self.output_dir, _MCTA_TEMPLATE_DIR)
    self._model_cards_dir = os.path.join(self.output_dir, _MODEL_CARDS_DIR)
    self._store = mlmd_store
    if self._store:
      if not model_uri:
        raise ValueError('If `mlmd_store` is set, `model_uri` should be set.')
      models = self._store.get_artifacts_by_uri(model_uri)
      if not models:
        raise ValueError(f'"{model_uri}" cannot be found in the `mlmd_store`.')
      if len(models) > 1:
        logging.info(
            '%d artifacts are found with the `model_uri`="%s". '
            'The last one is used.', len(models), model_uri)
      self._artifact_with_model_uri = models[-1]

  def _write_file(self, path: Text, content: Text) -> None:
    """Write content to the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as f:
      f.write(content)

  def _read_file(self, path: Text) -> Text:
    """Read content from a path."""
    with open(path, 'r') as f:
      return f.read()

  def scaffold_assets(self) -> ModelCard:
    """Generates the model cards tookit assets.

    Model cards assets include the model card json file and customizable model
    card UI templates.

    An assets directory is created if one does not already exist.

    If the MCT is initialized with a `mlmd_store`, it further auto-populates
    the model cards properties as well as generating related plots such as model
    performance and data distributions.

    Returns:
      A ModelCard representing the given model.

    Raises:
      FileNotFoundError: if it failed to copy the UI template files.
    """
    model_card = ModelCard()
    if self._store:
      model_card = tfx_util.generate_model_card_for_model(
          self._store, self._artifact_with_model_uri.id)
      metrics_artifacts = tfx_util.get_metrics_artifacts_for_model(
          self._store, self._artifact_with_model_uri.id)
      stats_artifacts = tfx_util.get_stats_artifacts_for_model(
          self._store, self._artifact_with_model_uri.id)

      for metrics_artifact in metrics_artifacts:
        eval_result = tfx_util.read_metrics_eval_result(metrics_artifact.uri)
        if eval_result is not None:
          graphics.annotate_eval_result_plots(model_card, eval_result)

      for stats_artifact in stats_artifacts:
        train_stats = tfx_util.read_stats_proto(stats_artifact.uri, 'train')
        eval_stats = tfx_util.read_stats_proto(stats_artifact.uri, 'eval')
        graphics.annotate_dataset_feature_statistics_plots(
            model_card, train_stats, eval_stats)

    # Write JSON file.
    self._write_file(self._mcta_json_file, model_card.to_json())

    # Write UI template files.
    for template_path in _UI_TEMPLATES:
      template_content = pkgutil.get_data('model_card_toolkit', template_path)
      if template_content is None:
        raise FileNotFoundError(f"Cannot find file: '{template_path}'")
      template_content = template_content.decode('utf8')
      self._write_file(
          os.path.join(self.output_dir, template_path), template_content)

    return model_card

  def update_model_card_json(self, model_card: ModelCard) -> None:
    """Validates the model card and updates the JSON file in MCT assets.

    If model_card.schema_version is not provided, it will assign the latest
    schema version to the `model_card`, and validate it.

    Args:
      model_card: The updated model card that users want to write back.

    Raises:
       Error: when the given model_card is invalid w.r.t. the schema.
    """
    if not model_card.schema_version:
      model_card.schema_version = validation.get_latest_schema_version()
    validation.validate_json_schema(model_card.to_dict(),
                                    model_card.schema_version)
    self._write_file(self._mcta_json_file, model_card.to_json())

  def export_format(self,
                    template_path: Text = None,
                    output_file=_DEFAULT_MODEL_CARD_FILE_NAME) -> Text:
    """Generates a model card based on the MCT assets.

    Args:
      template_path: The file path of the UI template. If not provided, the
        default UI template will be used.
      output_file: The file name of the generated model card. If not provided,
        the default 'model_card.html' will be used. If the file already exists,
        then it will be overwritten.

    Returns:
      The model card UI.
    """
    if not template_path:
      template_path = os.path.join(self._mcta_template_dir,
                                   _DEFAULT_UI_TEMPLATE_FILE)

    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    # Read JSON file.
    model_card = json.loads(self._read_file(self._mcta_json_file))

    # Generate Model Card.
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=True,
        auto_reload=True,
        cache_size=0)
    template = jinja_env.get_template(template_file)
    # TODO(b/154990170) Think about how to adjust the img inside template.
    model_card_file_content = template.render(
        model_details=model_card['model_details'],
        model_parameters=model_card.get('model_parameters', {}),
        quantitative_analysis=model_card.get('quantitative_analysis', {}),
        considerations=model_card.get('considerations', {}))

    # Write the model card file.
    mode_card_file_path = os.path.join(self._model_cards_dir, output_file)
    self._write_file(mode_card_file_path, model_card_file_content)

    return model_card_file_content

  def save_mlmd(self) -> None:
    """Saves the model card of the model artifact with `model_uri` to MLMD."""
    pass
