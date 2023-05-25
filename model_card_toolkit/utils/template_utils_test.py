# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for model_card_toolkit.utils.template_utils."""

import os
import tempfile

from absl.testing import absltest

from model_card_toolkit.utils import io_utils, template_utils


class TemplateUtilsTest(absltest.TestCase):
  def test_render(self):
    with tempfile.TemporaryDirectory() as test_dir:
      template_path = os.path.join(test_dir, 'test.txt.jinja')
      io_utils.write_file(template_path, '{{ greeting }}, World!')
      output_path = os.path.join(test_dir, 'test.txt')
      content = template_utils.render(
          template_path=template_path, output_path=output_path,
          template_variables={'greeting': 'Hello'}
      )
      read_content = io_utils.read_file(output_path)
      self.assertTrue(os.path.exists(output_path))
      self.assertEqual(content, read_content)
      self.assertEqual(content, 'Hello, World!')


if __name__ == '__main__':
  absltest.main()
