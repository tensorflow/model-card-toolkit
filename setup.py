# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py>=0.9,<0.11',
    'semantic-version>=2.8.0,<3',
    'jinja2>=2.10,<3',
    'matplotlib>=3.2.0,<4',
    'jsonschema>=3.2.0,<4',
    'tensorflow-data-validation>=0.26.0,<0.27.0',
    'tensorflow-model-analysis>=0.26.0,<0.27.0',
    'tensorflow-metadata>=0.26.0,<0.27.0',
    'ml-metadata>=0.26.0,<0.27.0',
    'dataclasses;python_version<"3.7"',
]

# Get version from version module.
with open('model_card_toolkit/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
  _LONG_DESCRIPTION = fh.read()

setup(
    name='model-card-toolkit',
    version=__version__,
    description='Model Card Toolkit',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/tensorflow/model-card-toolkit',
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    packages=find_packages(),
    package_data={
        'model_card_toolkit': [
            'schema/**/*.json', 'template/**/*.jinja'
        ]
    },
    python_requires='>=3.6,<4',
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='model card toolkit ml metadata machine learning',
)
