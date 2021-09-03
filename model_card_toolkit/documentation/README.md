# Model Card Toolkit





<g3mark-stacks product-id="model_card_toolkit" widget-kind="educational" use-name-in-title="true"></g3mark-stacks>



## Getting Started

```
import model_card_toolkit

# Initialize the Model Card Toolkit with a path to store generate assets
model_card_output_path = ...
mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

# Initialize the model_card_toolkit.ModelCard, which can be freely populated
model_card = mct.scaffold_assets()
model_card.model_details.name = 'My Model'

# Write the model card data to a file
mct.update_model_card(model_card)       # writes to proto
mct.update_model_card_json(model_card)  # writes to JSON

# Return the model card document as an HTML page
html = mct.export_format()
```

## Tutorials

*   [Standalone Model Card Toolkit](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/Standalone_Model_Card_Toolkit_Demo.ipynb)
*   [Scikit-Learn with Model Card Toolkit](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/Scikit_Learn_Model_Card_Toolkit_Demo.ipynb)
*   [MLMD with Model Card Toolkit](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/MLMD_Model_Card_Toolkit_Demo.ipynb)

If you want to try out the Model Card Toolkit (MCT) right away, you can run the
[standalone Model Card Toolkit demo](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/Standalone_Model_Card_Toolkit_Demo.ipynb),
or the
[Scikit-Learn Model Card Toolkit demo](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/Scikit_Learn_Model_Card_Toolkit_Demo.ipynb),
which demonstrates how MCT can be used in a Scikit-Learn workspace. If you are
using [MLMD/TFX](https://www.tensorflow.org/tfx) and want to incorporate MCT
into your workflow, you can try out the
[MLMD Model Card Toolkit demo](https://colab.sandbox.google.com/github/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/MLMD_Model_Card_Toolkit_Demo.ipynb).

These demos can be run directly from your browser. Click
[here](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/examples/README.md)
to learn more.

## Guides

The
[Concepts guide](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/guide/concepts.md)
provides a high-level explanation of the terminology used by MCT. Use this as a
reference as you read other documentation pages.

The
[Templates page](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/guide/templates.md)
explains how MCT uses templates as a skeleton to generate Model Card reports.
You can use a premade template provided by MCT, or you can create your own
template to generate Model Cards for your specific use case.
