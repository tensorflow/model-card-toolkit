"""Generates API docs for TensorFlow Model Remediation."""

import os

from absl import app
from absl import flags

import model_card_toolkit

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

FLAGS = flags.FLAGS

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default="/tmp/model_card_toolkit",
    help="Where to output the docs")

_CODE_URL_PREFIX = flags.DEFINE_string(
    "code_url_prefix",
    default="https://github.com/tensorflow/model-card-toolkit/tree/master/model-card-toolkit",
    help="The URL prefix for links to code.")

_SEARCH_HINTS = flags.DEFINE_bool(
    "search_hints",
    default=True,
    help="Include metadata search hints in the generated files")

_SITE_PATH = flags.DEFINE_string(
    "site_path",
    default="responsible-ai/model_card_toolkit/api_docs/python",
    help="Path prefix in the _toc.yaml")


def execute(output_dir: str, code_url_prefix: str, search_hints: bool,
            site_path: str):
  """Builds API docs for Model Card Toolkit."""

  # TODO(b/175031010): add missing class vars, or remove all class vars

  doc_generator = generate_lib.DocGenerator(
      root_title="Model Card Toolkit",
      py_modules=[("model_card_toolkit", model_card_toolkit)],
      base_dir=os.path.dirname(model_card_toolkit.__file__),
      search_hints=search_hints,
      code_url_prefix=code_url_prefix,
      site_path=site_path,
      callbacks=[
          public_api.explicit_package_contents_filter,
          public_api.local_definitions_filter
      ])

  doc_generator.build(output_dir)


def main(unused_argv):
  execute(_OUTPUT_DIR.value, _CODE_URL_PREFIX.value, _SEARCH_HINTS.value,
          _SITE_PATH.value)


if __name__ == "__main__":
  app.run(main)
