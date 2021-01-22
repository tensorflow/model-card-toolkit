# Model Card Toolkit

The Model Card Toolkit (MCT) streamlines and automates generation of [Model Cards](https://modelcards.withgoogle.com/about) [1], machine learning documents that provide context and transparency into a model's development and performance. Integrating the MCT into your ML pipeline enables the sharing model metadata and metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

![Generated model card image](https://raw.githubusercontent.com/tensorflow/model-card-toolkit/master/model_card_toolkit/documentation/guide/images/model_card.png)

## Installation

The Model Card Toolkit is hosted on [PyPI](https://pypi.org/project/model-card-toolkit/), and can be installed with `pip install model-card-toolkit` (or `pip install model-card-toolkit
--use-deprecated=legacy-resolver` for pip20.3). See [the installation guide](model_card_toolkit/documentation/guide/install.md) for more details.

## Getting Started

    import model_card_toolkit

    # Initialize the Model Card Toolkit with a path to store generate assets
    model_card_output_path = ...
    mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

    # Initialize the model_card_toolkit.ModelCard, which can be freely populated
    model_card = mct.scaffold_assets()
    model_card.model_details.name = 'My Model'

    # Write the model card data to a JSON file
    mct.update_model_card_json(model_card)

    # Return the model card document as an HTML page
    html = mct.export_format()

## Automatic Model Card Generation

If your machine learning pipeline uses the [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) platform or [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd), you can automate model card generation. See [this demo notebook](model_card_toolkit/documentation/examples/MLMD_Model_Card_Toolkit_Demo.ipynb) for a demonstration of how to integrate the MCT into your pipeline.

## Schema

Model cards are stored in JSON as an intermediate format. You can see the model card JSON schema in the `schema` directory. Note that this is not a finalized path and may be hosted elsewhere in the future.

## References

[1] https://arxiv.org/abs/1810.03993
