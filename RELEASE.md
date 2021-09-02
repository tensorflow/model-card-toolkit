<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (Still in Development)

## Major Features and Improvements

## Bug fixes and other changes

## Breaking changes and Deprecations

# Release 1.1.0

## Major Features and Improvements

## Bug fixes and other changes

* Update TFX compatibility to TFX 1.2.
* Fix bug where all datasets from MLMD were being compressed into one model_card.Dataset object.

## Breaking changes and Deprecations

# Release 1.0.0

## Major Features and Improvements

* Introduce `model_card.proto`. See https://developers.google.com/protocol-buffers for more info.
* All classes in `model_card_toolkit.model_card` submodule now have `to_proto()`, `merge_from_proto()`, `copy_from_proto()`, and `clear()` functions.
* `ModelCardToolkit.export_format()` now accepts `model_card` arg.
* `json_util.update()`, which updates a v0.0.1 JSON object to a v0.0.2 JSON object.

## Bug fixes and other changes

* Update default template layout so charts can wrap to multiple rows
* Installing from source now requires [Bazel](https://docs.bazel.build/versions/master/install.html)>=2.0.0.
* Update model card templates to use new schema.
* `model_card_toolkit.utils.validation.validate_json_schema()` can now validate both schema v0.0.1 and v0.0.2.
* Add `_jinja_loader` attribute to `model_card_toolkit` to allow custom Jinja loaders.

## Breaking changes

* JSON schema v0.0.2 replaces JSON schema v0.0.1.
* `ModelCardToolkit.update_model_card_json()` deprecated and replaced with `ModelCardToolkit.update_model_card()`. Writes to `data/model_card.proto` instead of `data/model_card.json`.
* `graphics.annotate_dataset_feature_statistics_plots` accepts a list of stats files instead of two individual train and eval stats params.

## Deprecations

# Release 0.1.3

## Major Features and Improvements

## Bug fixes and other changes

* Update model_card.py docstrings. Now hosted on [Responsible AI](https://www.tensorflow.org/responsible_ai/model_card_toolkit/api_docs/python/model_card_toolkit).

## Breaking changes

## Deprecations

# Release 0.1.2

## Major Features and Improvements

* In default_template.md.jinja and default_template.html.jinja, generate metrics table from `quantitative_analysis.performance_metrics`.

## Bug fixes and other changes

* Reference URLs in default HTML and Markdown template are now hyperlinks
* Fix bug where Considerations div is displayed in HTML model cards, even if Considerations div is empty.
* Update required fields in schema.
  * Removed considerations as required field.
  * Add lower_bound and upper_bound as required fields to confidence_interval.
* Fixed the part dependencies error for [new pip dependency resolver](https://pip.pypa.io/en/stable/user_guide/#changes-to-the-pip-dependency-resolver-in-20-3-2020).
* Update how UI templates are copied to be compatible with different platforms (colab, wetlab).
* Add model_card_toolkit.validation.validate_json_schema(), a function to validate a Python dictionary against the Model Card JSON schema.
* Fix the bug that some slices may have extra metrics that other slices does not have. e.g. __ERROR__ metric.

## Breaking changes

## Deprecations

# Release 0.1.1

## Major Features and Improvements

* add Markdown template

## Bug fixes and other changes

* remove `quantitative_analysis` from required fields
* add `input_format` and `export_format` fields
* add `model_architecture`, `input_format`, and `export_format` to HTML template
* add Cats vs Dogs util for `Standalone_Model_Card_toolkit_Demo.ipynb`

## Breaking changes

* Rename `_figure_to_base64str` to `figure_to_base64str`

## Deprecations

# Release 0.1.0

## Major Features and Improvements

Initial release of Model Card Toolkit.

## Bug fixes and other changes

## Breaking changes

## Deprecations
