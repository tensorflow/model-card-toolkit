# Install Model Card Toolkit

## Installing with pip

The Model Card Toolkit is hosted on
[PyPI](https://pypi.org/project/model-card-toolkit/), and requires Python 3.6 or
later.

```posix-terminal
pip install model-card-toolkit
```

You may need to append the `--use-deprecated=legacy-resolver` flag when running
versions of pip starting with 20.3

## Installing from source

Starting with
[version 0.1.4](https://github.com/tensorflow/model-card-toolkit/blob/master/RELEASE.md)
compiling Model Card Toolkit requires
[Bazel](https://docs.bazel.build/versions/master/install.html)>=2.0.0.

First, clone the github repo:

```posix-terminal
git clone https://github.com/tensorflow/model-card-toolkit.git
```

Build the pip package from source:

```posix-terminal
pip install wheel

cd model_card_toolkit

python3 setup.py sdist bdist_wheel
```

Finally, install your locally built package:

```posix-terminal
pip install --upgrade ./dist/*pkg.whl
```

## Filing a Bug

If you run into issues with Model Card Toolkit installation, please
[file an issue](https://github.com/tensorflow/model-card-toolkit/issues/new)
with details on your operating system version, Python version, pip version, and
locally-installed packages. You can find your locally-installed packages with
[`pip freeze`](https://pip.pypa.io/en/stable/reference/pip_freeze/)).
