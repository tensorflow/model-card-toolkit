# Model Card templates

[Jinja templates](https://jinja.palletsprojects.com/) are the backend structure of a Model Card document. The Model Card Toolkit comes with a few pre-made templates, but you can freely modify these templates or even build your own. In this document, we will discuss how to do this.

The following is the standard way you may initialize the Model Card Toolkit.

```py
mct_directory = ...  # where the Model Card assets will be generated
toolkit = ModelCardToolkit(mct_directory)
model_card = toolkit.scaffold_assets()
... # set the model_card's fields here
toolkit.update_model_card(model_card)
```

When you run `toolkit.scaffold_assets()`, the contents of [model_card_toolkit/template](https://github.com/tensorflow/model-card-toolkit/tree/main/model_card_toolkit/template) are copied into `mct_directory/template`. This includes premade templates such as [default_template.html.jinja](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/template/html/default_template.html.jinja) and [default_template.md.jinja](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/template/md/default_template.md.jinja).

The
[`model_card`](https://www.tensorflow.org/responsible_ai/model_card_toolkit/api_docs/python/model_card_toolkit/ModelCard)
object generated above can be manually populated. Once you are ready to generate
a Model Card document, you can pass the `model_card` back into MCT with
`toolkit.update_model_card(model_card)`.

### Use a Premade Model Card Template

We can then generate a Model Card document using one of the default templates, via the code below.

```py
template_path = os.path.join(mct_directory, 'template/html/default_template.html.jinja')
toolkit.export_format(template_path=template_path, output_file='model_card.html')
```

### Modify the Model Card Template

You can freely modify a premade template to change styling, reorganize information, etc. You should be familiar with the [Jinja API](https://jinja.palletsprojects.com/en/2.11.x/api/) and [control structures](https://jinja.palletsprojects.com/en/2.11.x/templates/#list-of-control-structures). Model Card field names are taken from [model_card.py](https://github.com/tensorflow/model-card-toolkit/blob/main/model_card_toolkit/model_card.py).

### Creating a new Model Card Template

Creating a new Model Card template works the same as modifying an existing one.

```py
my_custom_template_path = ...  # where the template is stored
toolkit.export_format(template_path=my_custom_template_path, output_file'model_card.html')  # generate the final Model Card
```
