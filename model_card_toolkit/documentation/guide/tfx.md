# Model Cards in TFX







The ModelCardGenerator TFX pipeline component generates model cards.

For the detailed model card format, see the
[Model Card API](https://www.tensorflow.org/responsible_ai/model_card_toolkit/api_docs/python/model_card_toolkit/ModelCard).

For more general information about TFX, please see the
[TFX User Guide](https://www.tensorflow.org/tfx/guide).

## Configuring the ModelCardGenerator Component

The ModelCardGenerator takes
[dataset statistics](https://www.tensorflow.org/tfx/guide/statsgen),
[model evaluation](https://www.tensorflow.org/tfx/guide/evaluator), and a
[pushed model](https://www.tensorflow.org/tfx/guide/pusher) to automatically
populate parts of a model card.

[Model card fields](https://www.tensorflow.org/responsible_ai/model_card_toolkit/api_docs/python/model_card_toolkit/ModelCard)
can also be explicitly populated with a JSON string (this can be generated using
the [`json`](https://docs.python.org/3/library/json.html) module, see Example
below). If a field is populated both by TFX and JSON, the JSON value will
overwrite the TFX value.

The ModelCardGenerator writes model card documents to the `model_card/`
directory of its artifact output. It uses a default HTML model card template,
which is used to generate `model_card.html`. Custom
[templates](https://www.tensorflow.org/responsible_ai/model_card_toolkit/guide/templates)
can also be used; each template input must be accompanied by a file name output
in the `template_io` arg.

### Example

```py
from model_card_toolkit import ModelCardGenerator
import json

...
model_card_fields = {
  'model_details': {
    'name': 'my_model',
    'owners': 'Google',
    'version': 'v0.1'
  },
  'considerations': {
    'limitations': 'This is a demo model.'
  }
}
mc_gen = ModelCardGenerator(
    statistics=statistics_gen.outputs['statistics'],
    evaluation=evaluator.outputs['evaluation'],
    pushed_model=pusher.outputs['pushed_model'],
    json=json.dumps(model_card_fields),
    template_io=[
        ('html/default_template.html.jinja', 'model_card.html'),
        ('md/default_template.md.jinja', 'model_card.md')
    ]
)
```

More details are available in the
[ModelCardGenerator](https://www.tensorflow.org/responsible_ai/model_card_toolkit/api_docs/python/model_card_toolkit/ModelCardGenerator)
API reference.
