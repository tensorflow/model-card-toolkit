# Model Card Toolkit

The Model Card Toolkit (MCT) streamlines and automates generation of [Model Cards](https://modelcards.withgoogle.com/about) [1], machine learning documents that provide context and transparency into a model's development and performance. Integrating the MCT into your ML pipeline enables the sharing model metadata and metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

![Generated model card image](https://raw.githubusercontent.com/tensorflow/model-card-toolkit/main/model_card_toolkit/documentation/guide/images/model_card.png)

## Installation

The Model Card Toolkit is hosted on [PyPI](https://pypi.org/project/model-card-toolkit/), and can be installed with `pip install model-card-toolkit` (or `pip install model-card-toolkit
--use-deprecated=legacy-resolver` for versions of pip starting with 20.3). See [the installation guide](model_card_toolkit/documentation/guide/install.md) for more details.

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

The `ModelCardGenerator` component is moving to the
[TFX Addons](https://github.com/tensorflow/tfx-addons) library and will no longer
be packaged in Model Card Toolkit from version 2.0.0. Before you can use the
component, you will need to install the `tfx-addons` package:

```sh
pip install tfx-addons[model_card_generator]
```

This page will be updated to include the new links for the Model Cards in TFX
guide and the end-to-end demo when the migration is completed.

## Schema

Model cards are stored in proto as an intermediate format. You can see the model card JSON schema in the `schema` directory.

## References

[1] https://arxiv.org/abs/1810.03993
