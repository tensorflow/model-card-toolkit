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
"""Framework-agnostic tests for model_card_toolkit.core."""

import os

from absl import flags
from absl.testing import absltest

from model_card_toolkit import core
from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.utils import io_utils


class CoreTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    test_dir = self.create_tempdir()
    self.mct_dir = test_dir.mkdir(
        os.path.join(test_dir, 'model_card')
    ).full_path

  def test_scaffold_assets(self):
    output_dir = self.mct_dir
    mct = core.ModelCardToolkit(output_dir=output_dir)
    self.assertEqual(mct.output_dir, output_dir)
    mct.scaffold_assets()
    self.assertIn(
        'default_template.html.jinja',
        os.listdir(os.path.join(output_dir, 'template/html'))
    )
    self.assertIn(
        'default_template.md.jinja',
        os.listdir(os.path.join(output_dir, 'template/md'))
    )
    self.assertIn(
        'model_card.proto', os.listdir(os.path.join(output_dir, 'data'))
    )

  def test_scaffold_assets_with_json(self):
    mct = core.ModelCardToolkit(output_dir=self.mct_dir)
    mc = mct.scaffold_assets({'model_details': {
        'name': 'json_test',
    }})
    self.assertEqual(mc.model_details.name, 'json_test')

  def test_update_model_card_with_valid_model_card(self):
    mct = core.ModelCardToolkit(output_dir=self.mct_dir)
    valid_model_card = mct.scaffold_assets()
    valid_model_card.model_details.name = 'My Model'
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')

    model_card_proto = io_utils.parse_proto_file(
        proto_path, model_card_pb2.ModelCard()
    )
    self.assertEqual(model_card_proto, valid_model_card.to_proto())

  def test_update_model_card_with_valid_model_card_as_proto(self):
    valid_model_card = model_card_pb2.ModelCard()
    valid_model_card.model_details.name = 'My Model'

    mct = core.ModelCardToolkit(output_dir=self.mct_dir)
    mct.update_model_card(valid_model_card)
    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')

    model_card_proto = io_utils.parse_proto_file(
        proto_path, model_card_pb2.ModelCard()
    )
    self.assertEqual(model_card_proto, valid_model_card)

  def test_export_format(self):
    mct = core.ModelCardToolkit(output_dir=self.mct_dir)
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)
    result = mct.export_format()

    proto_path = os.path.join(self.mct_dir, 'data/model_card.proto')
    self.assertTrue(os.path.exists(proto_path))
    model_card_proto = io_utils.parse_proto_file(
        proto_path, model_card_pb2.ModelCard()
    )
    self.assertEqual(model_card_proto.model_details.name, 'My Model')

    model_card_path = os.path.join(self.mct_dir, 'model_cards/model_card.html')
    self.assertTrue(os.path.exists(model_card_path))
    content = io_utils.read_file(model_card_path)
    self.assertEqual(content, result)
    self.assertTrue(content.startswith('<!DOCTYPE html>'))
    self.assertIn('My Model', content)

  def test_export_format_with_customized_template_and_output_name(self):
    mct = core.ModelCardToolkit(output_dir=self.mct_dir)
    mc = mct.scaffold_assets()
    mc.model_details.name = 'My Model'
    mct.update_model_card(mc)

    template_path = os.path.join(
        self.mct_dir, 'template/html/default_template.html.jinja'
    )
    output_file = 'my_model_card.html'
    result = mct.export_format(
        template_path=template_path, output_file=output_file
    )

    model_card_path = os.path.join(self.mct_dir, 'model_cards', output_file)
    self.assertTrue(os.path.exists(model_card_path))
    content = io_utils.read_file(model_card_path)
    self.assertEqual(content, result)
    self.assertTrue(content.startswith('<!DOCTYPE html>'))
    self.assertIn('My Model', content)

  def test_export_format_before_scaffold_assets(self):
    with self.assertRaises(ValueError):
      core.ModelCardToolkit().export_format()


if __name__ == '__main__':
  absltest.main()
else:
  # Manually pass and parse flags to prevent UnparsedFlagAccessError when using
  # pytest or unittest as a runner.
  flags.FLAGS(['--test_tmpdir'])
