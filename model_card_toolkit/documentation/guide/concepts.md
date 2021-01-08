# Model Card Toolkit Concepts



## Model Card

[Model Cards](https://arxiv.org/abs/1810.03993) are machine learning documents that provide context and transparency into a model's development and performance. They can be used to share model metadata and metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

### Schema

The [Model Card schema](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/schema) is a [JSON schema](https://json-schema.org/) describing a model card's available fields. These JSON objects can be interfaced with other systems for storage, analysis, or visualization.

#### Graphics

The `graphic.image` field is encoded as a [base64-encoded string](https://en.wikipedia.org/wiki/Base64). The Model Card Toolkit can help with generating base64 images - see [Model Card API](###model-card-api).

## Model Card Toolkit

The [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card_toolkit.py) allows you to generate [Model Card documents](###model-card-documents) and [JSON objects](###schema) with a streamlined Python interface.

### Model Card API

The Model Card Toolkit includes a Model Card API consisting of a [Python class](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card.py). Updates made to a Model Card Python object are written to a Model Card JSON object.

#### Graphics

The `model_card_toolkit.utils.graphics.figure_to_base64str()` function can be used to convert Matplotlib figures to base64 strings.

### Model Card Documents

By default, the generated model card document is a HTML file based on [default_template.html.jinja](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/template/html/default_template.html.jinja). However, you can provide your own template file to generate model cards in `ModelCardToolkit.export_format()`. These template files can be any text-based format (HTML, Markdown, LaTeX, etc.).

### TFX and MLMD Integration

The Model Card Toolkit integrates with the [TensorFlow Extended](https://www.tensorflow.org/tfx) and [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) tools. A Metadata Store can be used during Model Card Toolkit initialization to pre-populate many model card fields and generate training and evaluation plots. See [this demonstration](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/MLMD_Model_Card_Toolkit_Demo.ipynb) for a detailed example.
