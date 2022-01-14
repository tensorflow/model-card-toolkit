"""The ModelCard TFX/MLMD artifact."""

import datetime
from model_card_toolkit import model_card as mc_types
from tfx.types.artifact import Artifact
import ml_metadata as mlmd
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2


class ModelCard(Artifact):
  TYPE_NAME = 'ModelCard'


def create_model_card_artifact(
    model_card: mc_types.ModelCard,
    uri: str,
    store: mlmd.MetadataStore) -> metadata_store_pb2.Artifact:
  """Generates a ModelCard artifact at the specified uri.

  The `name` field is generated from a model's name and version. A timestamp is
  appended to prevent duplicate artifacts.

  Args:
    model_card: The model card to generate the ModelCard artifact from.
    uri: The uri where model card assets are stored.
    store: The MLMD store where the ModelCard artifact type lives.

  Returns:
    An artifact for the ModelCard object.
  """

  def _put_model_card_artifact_type(store: mlmd.MetadataStore) -> int:
    """Check if ModelCard artifact type exists in store. If not, add it.

    Args:
      store: The MLMD store to check for ModelCard artifact type.

    Returns:
      The type id of the ModelCard artifact type.
    """
    try:
      return store.get_artifact_type(ModelCard.TYPE_NAME).id
    except errors.NotFoundError:
      return store.put_artifact_type(
          metadata_store_pb2.ArtifactType(name=ModelCard.TYPE_NAME))

  def _generate_artifact_name(model_details: mc_types.ModelDetails) -> str:
    """Generates an artifact name from a model's name and version.

    Appends a timestamp to prevent duplicate artifacts.

    Args:
      model_details: Used to extract model name and version.

    Returns:
      A unique name for a model card artifact, of format
        `<model_name>_<version>_<timestamp>`. If model name or version name are
        not set in `model_details`, they are omitted.
    """
    name_builder = []
    if model_details.name:
      name_builder.append(model_details.name)
    if model_details.version.name:
      name_builder.append(model_details.version.name)
    name_builder.append(datetime.datetime.now().strftime('%H:%M:%S'))
    return '_'.join(name_builder)

  return metadata_store_pb2.Artifact(
      type=ModelCard.TYPE_NAME,
      type_id=_put_model_card_artifact_type(store),
      uri=uri,
      name=_generate_artifact_name(model_card.model_details))
