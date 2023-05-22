# Model Card Toolkit Concepts

## Model Card

[Model Cards](https://arxiv.org/abs/1810.03993) are machine learning documents that provide context and transparency into a model's development and performance. They can be used to share model metadata and metrics with researchers, developers, reporters, and more.

Some use cases of model cards include:

* Facilitating the exchange of information between model builders and product developers.
* Informing users of ML models to make better-informed decisions about how to use them (or how not to use them).
* Providing model information required for effective public oversight and accountability.

### Schema

The
[Model Card schema](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/proto/model_card.proto)
is a [proto](https://developers.google.com/protocol-buffers) describing a model
card's available fields. A
[JSON interface](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/schema)
is also available. These objects can be interfaced with other systems for
storage, analysis, or visualization.

Today, the Model Card schema is strictly enforced. In Model Card Toolkit 2.0,
this schema restriction will be lifted.

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
[Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/model_card_toolkit.py)
allows you to generate [Model Card documents](###model-card-documents), as well
as [Proto and JSON objects](###schema), with a streamlined Python interface.

### Model Card API

The Model Card Toolkit includes a Model Card API consisting of a
[Python class](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/model_card.py).
Updates made to a Model Card Python object are written to a Model Card proto
object.

#### Graphics

The `model_card_toolkit.utils.graphics.figure_to_base64str()` function can be
used to convert graphics, such as Matplotlib figures, to base64 strings.

#### Saving and Loading Model Cards

If you've finished annotating your model card and would like to serialize it in JSON
or protobuf format, use the method `ModelCard.save()`.

```python

import model_card_toolkit as mct

model_card = mct.ModelCard()
model_card.model_details.name = 'Fine-tuned MobileNetV2 Model for Cats vs. Dogs'
model_card.save('model_cards/cats_vs_dogs.json')
```

If you'd like to restore and update a saved model card, use the function
`model_card_toolkit.model_card.load_model_card()`.

```python

import model_card_toolkit as mct

model_card = mct.load_model_card('model_cards/cats_vs_dogs.json')
model_card.model_details.licenses.append(mct.License(identifier='Apache-2.0'))
```

### Model Card Documents

By default, the generated model card document is a HTML file based on
[default_template.html.jinja](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/template/html/default_template.html.jinja).
However, you can also provide your own custom Jinja template. These templates
files can be any text-based format (HTML, Markdown, LaTeX, etc.). A
[Markdown template](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/template/md/default_template.md.jinja)
is provided as an example.

### TFX and MLMD Integration

The Model Card Toolkit integrates with the
[TensorFlow Extended](https://www.tensorflow.org/tfx) and
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) tools. A Metadata Store
can be used during Model Card Toolkit initialization to pre-populate many model
card fields and generate training and evaluation plots.

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
