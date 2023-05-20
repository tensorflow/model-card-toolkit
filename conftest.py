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
"""Pytest configuration."""

import fnmatch
import pathlib

import pytest

_REQUIRES_OPTIONAL_DEPS = ['**/tf_*_test.py']


# Adapted from pytest-error-for-skips
# https://github.com/jankatins/pytest-error-for-skips/blob/2.0.0/pytest_error_for_skips.py
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
  """Hook to treat skipped tests as failures."""
  outcome = yield
  rep = outcome.get_result()

  if item.config.getoption('--fail-if-skipped'):
    if rep.skipped and call.excinfo.errisinstance(pytest.skip.Exception):
      rep.outcome = 'failed'
      r = call.excinfo._getreprcrash()
      rep.longrepr = f'Fail skipped tests - {r.message}'


@pytest.hookimpl(hookwrapper=True)
def pytest_ignore_collect(
    collection_path: pathlib.Path, config: pytest.Config
):
  """Hook to ignore tests that require optional dependencies."""
  outcome = yield
  if config.getoption('--ignore-requires-optional-deps'):
    if any(
        fnmatch.fnmatch(str(collection_path), pattern)
        for pattern in _REQUIRES_OPTIONAL_DEPS
    ):
      outcome.force_result(True)


def pytest_addoption(parser: pytest.Parser):
  parser.addoption(
      '--fail-if-skipped', action='store_true', default=False,
      help='Treat skipped tests as failures.'
  )

  parser.addoption(
      '--ignore-requires-optional-deps', action='store_true', default=False,
      help='Ignore tests that require optional dependencies.'
  )
