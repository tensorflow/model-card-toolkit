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
"""Package dependencies for model-card-toolkit."""

import importlib
from typing import Dict, List

_VERSIONS = {
    'absl': 'absl-py>=0.9,<1.1',
    'isort': 'isort',
    'jinja2': 'jinja2>=3.1,<3.2',
    'jsonschema': 'jsonschema>=3.2.0,<4',
    'matplotlib': 'matplotlib>=3.2.0,<4',
    'ml_metadata': 'ml-metadata>=1.5.0,<2.0.0',
    'pre-commit': 'pre-commit',
    'protobuf': 'protobuf>=3.19.0,<4',
    'pylint': 'pylint',
    'pytest': 'pytest',
    'tensorflow_data_validation': 'tensorflow-data-validation>=1.5.0,<2.0.0',
    'tensorflow_datasets': 'tensorflow-datasets>=4.8.2',
    'tensorflow_docs':
    'tensorflow-docs @ git+https://github.com/tensorflow/docs',
    'tensorflow_metadata': 'tensorflow-metadata>=1.5.0,<2.0.0',
    'tensorflow_model_analysis': 'tensorflow-model-analysis>=0.36.0,<0.42.0',
    'yapf': 'yapf',
}

_REQUIRED_DEPS = [
    'jinja2',  # rendering model card templates
    'jsonschema',  # validating JSON schema
    'matplotlib',  # plotting
    'protobuf',  # working with model card protos
]

_DOCS_EXTRA_DEPS = ['absl', 'tensorflow_docs']

_EXAMPLES_EXTRA_DEPS = [
    # Required for model_card_toolkit.documentation.examples.cats_vs_dogs
    'tensorflow_datasets',
]

_TENSORFLOW_EXTRA_DEPS = [
    'ml_metadata',
    'tensorflow_data_validation',
    'tensorflow_metadata',
    'tensorflow_model_analysis',
]

_TEST_EXTRA_DEPS = ['absl', 'isort', 'pre-commit', 'pylint', 'pytest', 'yapf']

TENSORFLOW_EXTRA_IMPORT_ERROR_MSG = """
This functionaliy requires `tensorflow` extra dependencies but they were not
found in your environment. You can install them with:
```
pip install model-card-toolkit[tensorflow]
```
"""


def _make_deps_list(package_names: List[str]) -> List[str]:
  """Returns a list of dependencies with their constraints.

  Raises: ValueError if a `package_name` is not in the list of known dependencies.
  """
  deps = []
  for package_name in package_names:
    if package_name not in _VERSIONS:
      raise ValueError(
          f'Package {package_name} is not in the list of known dependencies: '
          f'{_VERSIONS.keys()}'
      )
    deps.append(_VERSIONS[package_name])
  return deps


def make_required_install_packages() -> List[str]:
  """Returns the list of required packages."""
  return _make_deps_list(_REQUIRED_DEPS)


def make_extra_packages_docs() -> List[str]:
  """Returns the list of packages needed for building documentation."""
  return _make_deps_list(_DOCS_EXTRA_DEPS)


def make_extra_packages_examples() -> List[str]:
  """Returns the list of packages needed for running examples."""
  return _make_deps_list(_EXAMPLES_EXTRA_DEPS)


def make_extra_packages_tensorflow() -> List[str]:
  """Returns the list of packages needed to use TensorFlow utils."""
  return _make_deps_list(_TENSORFLOW_EXTRA_DEPS)


def has_tensorflow_extra_deps() -> bool:
  """Returns True if all tensorflow extra dependencies are installed."""
  return all(importlib.util.find_spec(name) for name in _TENSORFLOW_EXTRA_DEPS)


def assert_tensorflow_extra_deps_installed():
  """Raises ImportError if tensorflow extra dependencies are not installed.
  """
  if not has_tensorflow_extra_deps():
    raise ImportError(TENSORFLOW_EXTRA_IMPORT_ERROR_MSG)


def make_extra_packages_test() -> List[str]:
  """Returns the list of packages needed for running tests."""
  return _make_deps_list(_TEST_EXTRA_DEPS)


def make_extra_packages_all() -> List[str]:
  """Returns the list of all optional packages."""
  return [
      *make_extra_packages_docs(),
      *make_extra_packages_examples(),
      *make_extra_packages_tensorflow(),
      *make_extra_packages_test(),
  ]


def make_required_extra_packages() -> Dict[str, List[str]]:
  """Returns the dict of required extra packages."""
  return {
      'docs': make_extra_packages_docs(),
      'examples': make_extra_packages_examples(),
      'tensorflow': make_extra_packages_tensorflow(),
      'test': make_extra_packages_test(),
      'all': make_extra_packages_all(),
  }
