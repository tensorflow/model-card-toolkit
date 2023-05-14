# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Setup to install the Model Card Toolkit."""

import platform
import shutil
import subprocess
# TODO(b/188859752): deprecate distutils
from distutils.command import build

from setuptools import Command, setup

# Get dependency lists.
with open('model_card_toolkit/dependencies.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
make_required_install_packages = globals_dict['make_required_install_packages']
make_required_extra_packages = globals_dict['make_required_extra_packages']
make_extra_packages_test = globals_dict['make_extra_packages_test']

# Get version from version module.
with open('model_card_toolkit/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get long description.
with open('README.md', 'r', encoding='utf-8') as fh:
  _LONG_DESCRIPTION = fh.read()


class _BuildCommand(build.build):
  """Build everything that is needed to install.

  This overrides the original distutils "build" command to to run bazel_build
  command before any sub_commands.

  build command is also invoked from bdist_wheel and install command, therefore
  this implementation covers the following commands:
    - pip install . (which invokes bdist_wheel)
    - python setup.py install (which invokes install command)
    - python setup.py bdist_wheel (which invokes bdist_wheel command)
  """

  # Add "bazel_build" command as the first sub_command of "build". Each
  # sub_command of "build" (e.g. "build_py", "build_extbaz", etc.) is executed
  # sequentially when running a "build" command, if the second item in the tuple
  # (predicate method) is evaluated to true.
  sub_commands = [
      ('bazel_build', lambda self: True),
  ] + build.build.sub_commands


class _BazelBuildCommand(Command):
  """Build Bazel artifacts and move generated files."""
  def initialize_options(self):
    pass

  def finalize_options(self):
    # verified with bazel 2.0.0, 3.0.0, and 4.0.0 via bazelisk
    self._bazel_cmd = shutil.which('bazel')
    if not self._bazel_cmd:
      self._bazel_cmd = shutil.which('bazelisk')
    if not self._bazel_cmd:
      raise RuntimeError(
          'Could not find "bazel" or "bazelisk" binary. Please visit '
          'https://docs.bazel.build/versions/main/install.html for '
          'installation instruction.'
      )
    self._additional_build_options = []
    if platform.system() == 'Darwin':  # see b/175182911 for context
      self._additional_build_options = ['--macos_minimum_os=10.9']

  def run(self):
    subprocess.check_call(
        [
            self._bazel_cmd, 'run', '--verbose_failures',
            *self._additional_build_options,
            '//proto_build:move_generated_files'
        ]
    )


setup(
    name='model-card-toolkit',
    version=__version__,
    description='Model Card Toolkit',
    author='The TensorFlow Authors',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/tensorflow/model-card-toolkit',
    project_urls={
        'Bug Tracker': 'https://github.com/tensorflow/model-card-toolkit/issues',
        'Documentation': 'https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide',
    },
    packages=[
        'model_card_toolkit',
        'model_card_toolkit.documentation',
        'model_card_toolkit.documentation.examples',
        'model_card_toolkit.proto',
        'model_card_toolkit.tfx',
        'model_card_toolkit.utils',
    ],
    package_data={
        'model_card_toolkit': ['schema/**/*.json', 'template/**/*.jinja']
    },
    python_requires='>=3.7,<4',
    install_requires=make_required_install_packages(),
    tests_require=make_extra_packages_test(),
    extras_require=make_required_extra_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=['model card toolkit', 'ml metadata', 'machine learning'],
    cmdclass={
        'build': _BuildCommand,
        'bazel_build': _BazelBuildCommand,
    }
)  # yapf: disable
