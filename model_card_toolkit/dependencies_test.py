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
"""Tests for model_card_toolkit.utils.dependencies."""

import importlib
from unittest import mock

from absl.testing import absltest

from model_card_toolkit import dependencies

_MODULE_SPEC = importlib.machinery.ModuleSpec('placeholder_module', object())
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)

_MOCK_TENSORFLOW_EXTRA_MODULES = {
    package: _MODULE
    for package in dependencies._TENSORFLOW_EXTRA_DEPS
}

_MOCK_TENSORFLOW_EXTRA_MISSING_DEP = {
    dependencies._TENSORFLOW_EXTRA_DEPS[0]: None,
}


class DependenciesTest(absltest.TestCase):
  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MODULES)
  def test_has_tensorflow_extra_deps(self):
    assert dependencies.has_tensorflow_extra_deps()

  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MISSING_DEP)
  def test_has_tensorflow_extra_deps_with_missing_dep(self):
    assert not dependencies.has_tensorflow_extra_deps()

  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MODULES)
  def test_assert_tensorflow_extra_deps_installed(self):
    dependencies.assert_tensorflow_extra_deps_installed()

  @mock.patch.dict('sys.modules', _MOCK_TENSORFLOW_EXTRA_MISSING_DEP)
  def test_assert_tensorflow_extra_deps_installed_fails(self):
    with self.assertRaisesWithLiteralMatch(
        ImportError, dependencies.TENSORFLOW_EXTRA_IMPORT_ERROR_MSG
    ):
      dependencies.assert_tensorflow_extra_deps_installed()


if __name__ == '__main__':
  absltest.main()
