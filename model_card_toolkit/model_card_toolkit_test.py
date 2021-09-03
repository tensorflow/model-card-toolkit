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
"""Tests for model_card_toolkit."""

import os
from unittest import mock
import uuid

from absl.testing import absltest

from model_card_toolkit import model_card_toolkit
from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.utils import graphics
from model_card_toolkit.utils.testdata import testdata_utils


class ModelCardToolkitTest(absltest.TestCase):

  def setUp(self):
    super(ModelCardToolkitTest, self).setUp()
    self.tmp_db_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    f'test_mlmd_{uuid.uuid4()}.db')
    self.tmpdir = os.path.join(absltest.get_default_test_tmpdir(),
                               f'model_card_{uuid.uuid4()}')
    if not os.path.exists(self.tmpdir):
      os.makedirs(self.tmpdir)

  def test_init_with_store_no_model_uri(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(
        ValueError, 'If `mlmd_store` is set, `model_uri` should be set.'):
      model_card_toolkit.ModelCardToolkit(
          output_dir=self.tmpdir, mlmd_store=store)

  def test_init_with_store_model_uri_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    unknown_model = 'unknown_model'
    with self.assertRaisesRegex(
        ValueError, f'"{unknown_model}" cannot be found in the `mlmd_store`'):
      model_card_toolkit.ModelCardToolkit(
          mlmd_store=store, model_uri=unknown_model)

  def test_scaffold_assets(self):
    output_dir = self.tmpdir
    mct = model_card_toolkit.ModelCardToolkit(output_dir=output_dir)
    self.assertEqual(mct.output_dir, output_dir)
    mc = mct.scaffold_assets()  # pylint: disable=unused-variable
    self.assertIn('default_template.html.jinja',
                  os.listdir(os.path.join(output_dir, 'template/html')))
    self.assertIn('default_template.md.jinja',
                  os.listdir(os.path.join(output_dir, 'template/md')))
    self.assertIn('model_card.proto',
                  os.listdir(os.path.join(output_dir, 'data')))

  @mock.patch.object(
      graphics, 'annotate_dataset_feature_statistics_plots', autospec=True)
  @mock.patch.object(graphics, 'annotate_eval_result_plots', autospec=True)
  def test_scaffold_assets_with_store(self, mock_annotate_data_stats,
                                      mock_annotate_eval_results):
    num_stat_artifacts = 2
    num_eval_artifacts = 1
    output_dir = self.tmpdir
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=output_dir,
        mlmd_store=store,
        model_uri=testdata_utils.TFX_0_21_MODEL_URI)
    mc = mct.scaffold_assets()
    self.assertIsNotNone(mc.model_details.name)
    self.assertIsNotNone(mc.model_details.version.name)
    self.assertIn('default_template.html.jinja',
                  os.listdir(os.path.join(output_dir, 'template/html')))
    self.assertIn('default_template.md.jinja',
                  os.listdir(os.path.join(output_dir, 'template/md')))
    self.assertEqual(mock_annotate_data_stats.call_count, num_stat_artifacts)
    self.assertEqual(mock_annotate_eval_results.call_count, num_eval_artifacts)

  def test_update_model_card_with_valid_model_card(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    valid_model_card = mct.scaffold_assets()
    valid_model_card.model_details.name = 'My Model'
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.tmpdir, 'data/model_card.proto')

    model_card_proto = model_card_pb2.ModelCard()
    with open(proto_path, 'rb') as f:
      model_card_proto.ParseFromString(f.read())
    self.assertEqual(model_card_proto, valid_model_card.to_proto())

  def test_export_format(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    mct = model_card_toolkit.ModelCardToolkit(
        output_dir=self.tmpdir,
        mlmd_store=store,
        model_uri=testdata_utils.TFX_0_21_MODEL_URI)
    model_card = mct.scaffold_assets()
    model_card.model_details.name = 'My Model'
    mct.update_model_card(model_card)
    result = mct.export_format()

    proto_path = os.path.join(self.tmpdir, 'data/model_card.proto')
    self.assertTrue(os.path.exists(proto_path))
    with open(proto_path, 'rb') as f:
      model_card_proto = model_card_pb2.ModelCard()
      model_card_proto.ParseFromString(f.read())
      self.assertEqual(model_card_proto.model_details.name, 'My Model')
    model_card_path = os.path.join(self.tmpdir, 'model_cards/model_card.html')
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_with_customized_template_and_output_name(self):
    mct = model_card_toolkit.ModelCardToolkit(output_dir=self.tmpdir)
    model_card = mct.scaffold_assets()
    model_card.model_details.name = 'My Model'
    mct.update_model_card(model_card)

    template_path = os.path.join(
        self.tmpdir, 'template/html/default_template.html.jinja')
    output_file = 'my_model_card.html'
    result = mct.export_format(
        template_path=template_path, output_file=output_file)

    model_card_path = os.path.join(self.tmpdir, 'model_cards', output_file)
    self.assertTrue(os.path.exists(model_card_path))
    with open(model_card_path) as f:
      content = f.read()
      self.assertEqual(content, result)
      self.assertTrue(content.startswith('<!DOCTYPE html>'))
      self.assertIn('My Model', content)

  def test_export_format_before_scaffold_assets(self):
    with self.assertRaises(ValueError):
      model_card_toolkit.ModelCardToolkit().export_format()


if __name__ == '__main__':
  absltest.main()
