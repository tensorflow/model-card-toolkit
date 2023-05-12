# Model Card Toolkit

[![CI][ci_badge]][ci_link]
[![PyPI][pypi_badge]][pypi_link]
[![Documentation][docs_badge]][docs_link]

The Model Card Toolkit (MCT) streamlines and automates generation of
[Model Cards](https://modelcards.withgoogle.com/about) [1], machine learning documents
that provide context and transparency into a model's development and performance.
Integrating the MCT into your ML pipeline enables you to share model metadata and
metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

![Generated model card image](https://raw.githubusercontent.com/tensorflow/model-card-toolkit/main/model_card_toolkit/documentation/guide/images/model_card.png)

## Installation

The Model Card Toolkit is hosted on [PyPI](https://pypi.org/project/model-card-toolkit/),
and requires Python 3.7 or later.

Installing the basic, framework agnostic package:

```sh
pip install model-card-toolkit
```

If you are generating model cards for TensorFlow models, install the optional
TensorFlow dependencies to use Model Card Toolkit's TensorFlow utilities:

```sh
pip install model-card-toolkit[tensorflow]
```

You may need to append the `--use-deprecated=legacy-resolver` flag when running
versions of pip starting with 20.3.

See [the installation guide](model_card_toolkit/documentation/guide/install.md)
for more installation options.

## Getting Started

    import model_card_toolkit

    # Initialize the Model Card Toolkit with a path to store generate assets
    model_card_output_path = ...
    mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

    # Initialize the model_card_toolkit.ModelCard, which can be freely populated
    model_card = mct.scaffold_assets()
    model_card.model_details.name = 'My Model'

    # Write the model card data to a proto file
    mct.update_model_card(model_card)

    # Return the model card document as an HTML page
    html = mct.export_format()

## Model Card Generation on TFX

If you are using [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), you can
incorporate model card generation into your TFX pipeline via the `ModelCardGenerator`
component.

The `ModelCardGenerator` component has moved to the
[TFX Addons](https://github.com/tensorflow/tfx-addons) library and is no longer
packaged in Model Card Toolkit from version 2.0.0. Before you can use the
component, you will need to install the `tfx-addons` package:

```sh
pip install tfx-addons[model_card_generator]
```

See the [ModelCardGenerator guide](https://github.com/tensorflow/tfx-addons/blob/main/tfx_addons/model_card_generator/README.md)
and run the [case study notebook](https://github.com/tensorflow/tfx-addons/blob/main/examples/model_card_generator/MLMD_Model_Card_Toolkit_Demo.ipynb)
to learn more about the component.

## Schema

Model cards are stored in proto as an intermediate format. You can see the model
card JSON schema in the `schema` directory.

## References

[1] https://arxiv.org/abs/1810.03993


[ci_badge]: https://github.com/tensorflow/model-card-toolkit/actions/workflows/ci.yml/badge.svg
[ci_link]: https://github.com/tensorflow/model-card-toolkit/actions/workflows/ci.yml

[pypi_badge]: https://badge.fury.io/py/model-card-toolkit.svg
[pypi_link]: https://badge.fury.io/py/model-card-toolkit

[docs_badge]: https://img.shields.io/badge/TensorFow-page-orange
[docs_link]: https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide
