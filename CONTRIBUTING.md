# How to Contribute

We'd love to accept your patches and contributions to the Model Card Toolkit and appreciate all kinds of help and are working to make this guide as comprehensive as possible. Please let us know if you think of something we could do to help lower the barrier to contributing.

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

***NOTE***: If you are interested in contributing no-code contributions like user studies, design reviews, UX studies around responsible machine learning, please read the section [How to contribute no-code submissions](#how-to-contribute-no-code-submissions)

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

## Contributing code

If you have improvements to the Model Card Toolkit, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

### Code Ownership

* Code ownership is tracked through the `CODEOWNERS` file. Users can be added if one of the following situations apply:
  * Developers who contribute or maintain the Model Card Toolkit can gain coder owner access to the project.
  * Requesting project code ownership requires a substantial contribution (e.g. contributed to the bi-weekly meetings, maintained some code over some time, or contributed a major feature).


### Development tips

We use [pre-commit](https://pre-commit.com/) to validate our code before we push to the repository. We use push over commit to allow more flexibility.

Here's how to install it locally:
- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate`
- Upgrade pip: `pip install --upgrade pip`
- Install test packages: `pip install -e ".[test]"`
- Install pre-commit hooks for push hooks: `pre-commit install --hook-type pre-push`
- Change and commit files. pre-commit will run the tests and linting before you push. You can also manually trigger the tests and linting with `pre-commit run --hook-stage push --all-files`

Note that pre-commit will be run via GitHub Action automatically for new PRs.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/tensorflow/model-card-toolkit/pulls),
make sure your changes are consistent with the guidelines and follow our coding style.

#### General guidelines and philosophy for contribution

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and b) guard against future breaking
  changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.

#### Python coding style

Changes to Python code should conform to
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with indent width of 2 spaces.

This is enforced using [pre-commit](https://pre-commit.com/) hooks that run: `yapf`, `isort`, `pylint`.

To run the checks manually, follow [Development tips](#development-tips) and run:
```bash
pre-commit run --hook-stage push --files model-card-toolkit/__init__.py
```

#### License

Include a license at the top of new files.

* [Python license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py#L1)

#### Testing your code

We use pytest to run tests. You can run tests locally using:

- Create virtual environemnt: `python3 -m venv env`
- Activate virtual environment: `source env/bin/activate && pip install --upgrade pip`
- Choose component to develop: `export COMPONENT_NAME=mlmd_client` (replace with the component you will be developing)
- Install test packages: `pip install -e ".[$COMPONENT_NAME,test]"`
- Run tests: `python -m pytest model-card-toolkit/`

Note that only files that end with `_test.py` will be recognized as test. Learn more on writing pytest tests in [pytest docs](https://docs.pytest.org/en/latest/getting-started.html#create-your-first-test).


## How to contribute no-code submissions

If you are a UX designer, responsible ML researcher or simply interested in the responsible use of machine learning, and you would like to contribute no-code submission like user studies, requirements research, model card designs, etc., you can either

* Create a Github discussion with your submission [here](https://github.com/tensorflow/model-card-toolkit/discussions)
* Join our bi-weekly project group meetings and join the discussion
