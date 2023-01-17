# Contribute to Model Card Toolkit

Model Card Toolkit can only grow through the contributions of this community.
Thanks so much for your enthusiasm and your work—we appreciate everything you do!

There are many ways to contribute to Model Card Toolkit! You can contribute code,
improve tests and API documentation, or create new end-to-end examples. Code
contributions are not the only way to help the community. Proposing usability
and design improvements, answering questions, and helping others are also
immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on social media every
time it has helped you, or simply star the repository to say thank you.

For more ways to contribute, refer to GitHub's
[guide to making open source contributions](https://opensource.guide/how-to-contribute/)

**However you choose to contribute, please follow
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).**

*This guide was heavily inspired by other guides in the
[Tensorflow GitHub organization](https://github.com/tensorflow) and the
[🤗 Transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).*

## Contributing Code

If you have improvements to Model Card Toolkit, send us your pull requests!
Before writing any code, we strongly advise you to search through the existing
[pull requests](https://github.com/tensorflow/model-card-toolkit/pulls)
or [issues](https://github.com/tensorflow/model-card-toolkit/issues) to make sure
nobody is already working on the same thing.

If you would like to contribute code and are not sure where to start, take look
at issues with the labels
["good first issue"](https://github.com/tensorflow/model-card-toolkit/labels/good%20first%20issue)
or ["contributions welcome"](https://github.com/tensorflow/model-card-toolkit/labels/stat%3Acontributions%20welcome).

When you find an issue that you want to work on, please leave a comment so other
people know you're working on it. If you need help or would prefer to work on the
issue with others, you may use the issue comment thread ask questions or coordinate.

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. For those just getting started, GitHub
has a [pull requests guide](https://help.github.com/articles/using-pull-requests/).

Before submitting your pull request for review, make sure your changes follow the
Model Card Toolkit guidelines and coding style. Maintainers and other contributors
will review your pull request. Please participate in the discussion and make the
requested changes.

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump
a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

- If you are an individual writing original source code and you're sure you own the
  intellectual property, then you'll need to sign an
  [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
- If you work for a company that wants to allow you to contribute your work, then you'll
  need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions
for how to sign and return it. Once we receive it, we'll be able to accept your pull
requests.

***NOTE***: Only original source code from you and other people that have signed the
CLA can be accepted into the main repository.

### Setting up a local development environment

Follow these steps to install `model-card-toolkit`:

1. Install [Bazel](https://bazel.build/install), which powers the protobuf stub
   code generation. Confirm that Bazel is installed and executable:

   ```sh
   bazel --version
   ```

2. Fork the [repository](https://github.com/tensorflow/model-card-toolkit) by
   clicking on the **[Fork](https://github.com/tensorflow/model-card-toolkit/fork)**
   button on the repository's page. This creates a copy of the code under your
   GitHub user account.

3. Clone your fork to your local machine:

   ```sh
   git clone git@github.com:<your GitHub username>/model-card-toolkit.git
   cd model-card-toolkit
   ```

4. Create and activate a virtual environment:

   ```sh
   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   ```

5. Install the `model-card-toolkit` development package in editable mode:

   ```sh
   pip install -e ".[test]"
   ```

   When you install the library in editable mode (with the `-e` flag), your local
   changes will be picked up without needing to re-install the library.

### Re-generating protobuf stub code

If you make changes to a `.proto` file, you should re-generate the
protobuf stub code before using it. The command used to do this is
automatically invoked once when you first install `model-card-toolkit` in
editable mode, but further stub generation requires manual invocation.

```sh
bazel run //model_card_toolkit:move_generated_files
```

### Licenses

Include a license at the top of new files.

- [Python license example](https://github.com/tensorflow/model-card-toolkit/blob/master/setup.py#L1)
- [Bash license example](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/move_generated_files.sh#L2)

Bazel BUILD files also need to include a license section. See
[BUILD example](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/BUILD#L15).

**Do not** include a license at the top of Jinja template files.

### Python coding style

Changes to Model Card Toolkit Python code should conform to the
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
with indentation width of 2 spaces.

### Linting your code

Please check your code for linting errors before submitting your PR for review. Pull requests are lint checked using `pre-commit` and `pylint`.
If you want to run the [`pre-commit`](https://pre-commit.com/) checks locally, please install

```sh
pip install pre-commit
```

When you have `pre-commit` installed, you can run follow command from your local project folder to check for linting errors.
```sh
pre-commit run --all-files
```

### Testing your changes

#### Writing and updating unit tests

Include unit tests when you contribute new features, as they help to a) prove
that your code works correctly, and b) guard against future breaking
changes to lower maintenance costs. Bug fixes also generally require creating
or updating unit tests, because the presence of bugs usually indicates insufficient
test coverage.

In general, all Python files have at least one corresponding test file. For example,
`awesome.py` should have a corresponding `awesome_test.py`.

#### Running unit tests

To run a specific test suite, e.g. `ModelCardTest`, run its test file:

```sh
pytest model_card_toolkit/model_card_test.py
```

Use the following command to run all unit tests:

```sh
pytest model_card_toolkit
```
