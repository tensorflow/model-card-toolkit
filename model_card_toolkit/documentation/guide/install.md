# Install Model Card Toolkit

## Installing with pip

The Model Card Toolkit is hosted on
[PyPI](https://pypi.org/project/model-card-toolkit/), and requires Python 3.7 or
later.

Installing the basic, framework agnostic package:

```sh
pip install model-card-toolkit
```

If you are generating model cards for TensorFlow models, install the optional TensorFlow dependencies to use Model Card Toolkit's TensorFlow utilities:

```sh
pip install model-card-toolkit[tensorflow]
```

You may need to append the `--use-deprecated=legacy-resolver` flag when running
versions of pip starting with 20.3 if the pip resolver is taking too long.

## Installing from source

Before you can install Model Card Toolkit from source, you need to install
[Bazel](https://bazel.build/install)>=2.0.0, which powers the protobuf stub code
generation.

Confirm that Bazel is installed and executable:

```sh
bazel --version
```

Clone the Model Card Toolkit repo from GitHub:

```sh
git clone https://github.com/tensorflow/model-card-toolkit.git
```

Build the pip package from source:

```sh
pip install wheel

python setup.py sdist bdist_wheel
```

Finally, install your locally built package:

```sh
pip install --upgrade ./dist/model_card_toolkit-*.whl
```

## Filing a Bug

If you run into issues with Model Card Toolkit installation, please
[file an issue](https://github.com/tensorflow/model-card-toolkit/issues/new)
with details on your operating system version, Python version, pip version, and
locally-installed packages. You can find your locally-installed packages with
[`pip freeze`](https://pip.pypa.io/en/stable/reference/pip_freeze/)).
