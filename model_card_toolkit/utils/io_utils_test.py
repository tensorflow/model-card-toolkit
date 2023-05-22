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
"""Tests for model_card_toolkit.utils.io_utils."""

import os
import tempfile

from absl.testing import absltest

from model_card_toolkit.proto import model_card_pb2
from model_card_toolkit.utils import io_utils


class IoUtilsTest(absltest.TestCase):
  def test_suffix(self):
    self.assertEqual('.json', io_utils.suffix('test.json'))
    self.assertEqual('.gz', io_utils.suffix('test.json.gz'))
    self.assertEqual('', io_utils.suffix('test'))
    self.assertEqual('', io_utils.suffix('.json'))

  def test_write_and_read_file(self):
    with tempfile.TemporaryDirectory() as test_dir:
      path = os.path.join(test_dir, 'test.txt')
      content = 'This is a sentence.'
      io_utils.write_file(path, content)
      read_content = io_utils.read_file(path)
      self.assertEqual(content, read_content)

  def test_write_and_parse_proto(self):
    with tempfile.TemporaryDirectory() as test_dir:
      path = os.path.join(test_dir, 'test.proto')
      proto = model_card_pb2.KeyVal(key='key', value='value')
      io_utils.write_proto_file(path, proto)
      parsed_proto = io_utils.parse_proto_file(path, model_card_pb2.KeyVal())
      self.assertEqual(proto, parsed_proto)
