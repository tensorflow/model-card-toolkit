<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version(Still in Development)

* Fixed the part dependencies error for [new pip dependency resolver](https://pip.pypa.io/en/stable/user_guide/#changes-to-the-pip-dependency-resolver-in-20-3-2020).

## Major Features and Improvements

## Bug fixes and other changes

* Reference URLs in default HTML and Markdown template are now hyperlinks
* Fix bug where Considerations div is displayed in HTML model cards, even if Considerations div is empty.
* Update required fields in schema.
  * Removed considerations as required field.
  * Add lower_bound and upper_bound as required fields to confidence_interval.

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
