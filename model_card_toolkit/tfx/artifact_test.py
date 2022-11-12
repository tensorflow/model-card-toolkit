"""Tests for artifact."""

from absl.testing import absltest
from model_card_toolkit.tfx import artifact
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class ArtifactTest(absltest.TestCase):

  def setUp(self):
    super(ArtifactTest, self).setUp()
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    self.store = mlmd.MetadataStore(connection_config)

  def test_create_and_save_artifact(self):
    mc_artifact = artifact.create_and_save_artifact(
        artifact_name='my model',
        artifact_uri='/path/to/model/card/assets',
        store=self.store)

    with self.subTest('saved_to_mlmd'):
      self.assertCountEqual([mc_artifact],
                            self.store.get_artifacts_by_id([mc_artifact.id]))
    with self.subTest('properties'):
      with self.subTest('type_id'):
        self.assertEqual(mc_artifact.type_id,
                         self.store.get_artifact_type('ModelCard').id)
      with self.subTest('uri'):
        self.assertEqual(mc_artifact.uri, '/path/to/model/card/assets')
      with self.subTest('name'):
        self.assertStartsWith(mc_artifact.name, 'my model_')


if __name__ == '__main__':
  absltest.main()
