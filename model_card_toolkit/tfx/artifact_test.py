"""Tests for artifact."""

from absl.testing import absltest
from absl.testing import parameterized
from model_card_toolkit import model_card
from model_card_toolkit.tfx import artifact
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_TYPE_NAME = 'ModelCard'


class ArtifactTest(parameterized.TestCase):

  def setUp(self):
    super(ArtifactTest, self).setUp()
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    self.store = mlmd.MetadataStore(connection_config)

  @parameterized.parameters([True, False])
  def test_put_artifact_type_and_create_artifact(self, version: bool):
    mc = model_card.ModelCard()
    mc.model_details.name = 'my model'
    if version:
      mc.model_details.version.name = 'v1'

    type_id = artifact.put_model_card_artifact_type(self.store)
    with self.subTest(name='put_model_card_artifact_type'):
      self.assertEqual(type_id,
                       artifact.put_model_card_artifact_type(self.store))

    model_card_assets_path = '/path/to/model/card/assets'
    mc_artifact = artifact.create_model_card_artifact(mc,
                                                      model_card_assets_path,
                                                      type_id)

    with self.subTest(name='create_model_card_artifact'):
      self.assertEqual(mc_artifact.type, _TYPE_NAME)
      self.assertEqual(mc_artifact.type_id,
                       self.store.get_artifact_type(_TYPE_NAME).id)
      self.assertEqual(mc_artifact.uri, model_card_assets_path)
      if version:
        self.assertStartsWith(mc_artifact.name, 'my model_v1_')
      else:
        self.assertStartsWith(mc_artifact.name, 'my model_')


if __name__ == '__main__':
  absltest.main()
