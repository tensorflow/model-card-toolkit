# Model Cards on TFX

TODO(b/210475018): use this documentation to develop the MCT-TFX demo

TODO(b/209051205): when publishing MCT v1.3.0, incorporate this into the MCT docs

## ModelCard Artifact

`model_card_toolkit.tfx.artifact.ModelCard` is a [TFX/MLMD Artifact](https://www.tensorflow.org/tfx/guide/mlmd#data_model) that stores the model card assets at a location specified by its `uri` property. These assets include...

* a data file containing the model card fields, located at `<uri>/data/model_card.proto`.
* the model card itself, located at the `<uri>/model_card/` directory.

## ModelCardGenerator

`model_card_toolkit.tfx.component.ModelCardGenerator` is a [TFX Component](https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines#component) that generates model cards from its inputs. The model cards assets are saved to a ModelCard artifact that can be fetched from the `outputs['model_card]'` property.

```python
context = InteractiveContext()
...
mc_gen = ModelCardGenerator(
    statistics=statistics_gen.outputs['statistics'],
    evaluation=evaluator.outputs['evaluation'],
    pushed_model=pusher.outputs['pushed_model'],
    json="{'model_details': {'name': 'my_model'}}",
    template_io=[
        ('html/default_template.html.jinja', 'model_card.html'),
        ('md/default_template.md.jinja', 'model_card.md')
    ]
    )
context.run(mc_gen)
mc_artifact = mc_gen.outputs['model_card']
mc_path = os.path.join(mc_artifact.uri, 'model_card', 'model_card.html')
with open(mc_path) as f:
  mc_content = f.readlines()
```

### Args

Note that all inputs are optional.

* **statistics**: The output of a [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) component. This contains dataset statistics that are used to populate the model card.
* **evaluation**: The output of a [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator) component. This contains model evaluations that are used to populate the model card.
* **pushed_model**: The output of a [Pusher](https://www.tensorflow.org/tfx/guide/pusher) component. This contains the pushed model used to populate the model card.
* **json**: A JSON string that corresponds to the [Model Card JSON schema](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/schema/v0.0.2/model_card.schema.json). This allows manual population of model card fields. Fields populated by JSON will overwrite values extracted from the TFX components above.
* **template_io**: A list of input/output pairs. The input is the path to a [Jinja](https://jinja.palletsprojects.com/) template. Using the data extracted from the above TFX components and JSON string, this template is populated and saved to generate a model card. The output is the filename where the model card is written in the `model_card/` directory. By default, the values `html/default_template.html.jinja` and `model_card.html` are used.
