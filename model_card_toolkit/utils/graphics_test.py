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
"""Tests for model_card_toolkit.utils.graphics."""

from absl.testing import absltest, parameterized

from model_card_toolkit.utils import graphics


class GraphicsTest(parameterized.TestCase):
  @parameterized.parameters(
      [
          [(), ('Overall', 'Overall')],
          [(('gender', 'male'), ), ('gender', 'male')],
          [
              (
                  ('gender', 'male'),
                  ('zip', 12345),
              ), ('gender, zip', 'male, 12345')
          ],
          [
              (
                  ('gender', 'male'),
                  ('zip', 12345),
                  ('height', 5.7),
              ), ('gender, zip, height', 'male, 12345, 5.7')
          ],
          [
              (
                  ('gender', 'male'), ('zip', 12345), ('height', 5.7),
                  ('comment', u'你好')
              ), ('gender, zip, height, comment', u'male, 12345, 5.7, 你好')
          ],
      ]
  )
  def test_stringify_slice_key(self, slices, expected_result):
    result = graphics.stringify_slice_key(slices)
    self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
