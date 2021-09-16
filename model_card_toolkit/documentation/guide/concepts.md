# Model Card Toolkit Concepts







## Model Card

[Model Cards](https://arxiv.org/abs/1810.03993) are machine learning documents that provide context and transparency into a model's development and performance. They can be used to share model metadata and metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

### Schema

The
[Model Card schema](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/proto/model_card.proto)
is a [proto](https://developers.google.com/protocol-buffers) describing a model
card's available fields. A
[JSON interface](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/schema)
is also available. These objects can be interfaced with other systems for
storage, analysis, or visualization.

#### Graphics

Model Card Toolkit automatically generates graphics for TFX datasets and
evaluation results. Graphics can also be manually created using a tool like
Matplotlib, and written to a ModelCard - see [Model Card API](###model-card-api)
for details.

In the Model Card schema, graphics are stored in the
[Graphic.image](https://github.com/tensorflow/model-card-toolkit/blob/3b565d9ec14dbf147756379649a3a32934921460/model_card_toolkit/model_card.py#L154)
field, and are encoded as
[base64-encoded strings](https://en.wikipedia.org/wiki/Base64). The Model Card
Toolkit can help with [generating base64 images](###model-card-api).

## Model Card Toolkit

The
[Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card_toolkit.py)
allows you to generate [Model Card documents](###model-card-documents), as well
as [Proto and JSON objects](###schema), with a streamlined Python interface.

### Model Card API

The Model Card Toolkit includes a Model Card API consisting of a
[Python class](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card.py).
Updates made to a Model Card Python object are written to a Model Card proto
object.

#### Graphics

The `model_card_toolkit.utils.graphics.figure_to_base64str()` function can be
used to convert graphics, such as Matplotlib figures, to base64 strings.

### Model Card Documents

By default, the generated model card document is a HTML file based on
[default_template.html.jinja](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/template/html/default_template.html.jinja).
A
[Markdown template](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/template/md/default_template.md.jinja)
is also provided. However, you can provide your own template file to generate
model cards via `ModelCardToolkit.export_format()`. These template files can be
any text-based format (HTML, Markdown, LaTeX, etc.).

### TFX and MLMD Integration

The Model Card Toolkit integrates with the [TensorFlow Extended](https://www.tensorflow.org/tfx) and [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) tools. A Metadata Store can be used during Model Card Toolkit initialization to pre-populate many model card fields and generate training and evaluation plots. See [this demonstration](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/MLMD_Model_Card_Toolkit_Demo.ipynb) for a detailed example.

[Artifacts](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_artifacts.py)
used by MCT:

*   [Examples](https://github.com/tensorflow/tfx/blob/74978506db5b7463c6f3c5b0716c4e834314b596/tfx/types/standard_artifacts.py#L76)
    and
    [ExampleStatistics](https://github.com/tensorflow/tfx/blob/74978506db5b7463c6f3c5b0716c4e834314b596/tfx/types/standard_artifacts.py#L93):
    used to plot slice count graphs for each dataset
    ([TFDV](https://www.tensorflow.org/tfx/data_validation/get_started)).
*   [Model](https://github.com/tensorflow/tfx/blob/74978506db5b7463c6f3c5b0716c4e834314b596/tfx/types/standard_artifacts.py#L114)
    and
    [ModelEvaluation](https://github.com/tensorflow/tfx/blob/74978506db5b7463c6f3c5b0716c4e834314b596/tfx/types/standard_artifacts.py#L126):
    used to plot
    [TFMA](https://www.tensorflow.org/tfx/model_analysis/get_started) sliced
    evaluation metrics.

[Executions](https://github.com/tensorflow/tfx/blob/master/tfx/types/standard_component_specs.py)
used by MCT:

*   [Trainer](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer):
    used to populate model name and version
