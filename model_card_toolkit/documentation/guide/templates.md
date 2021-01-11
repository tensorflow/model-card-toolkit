# Model Card templates







[Jinja templates](https://jinja.palletsprojects.com/) are the backend structure of a Model Card document. The Model Card Toolkit comes with a few pre-made templates, but you can freely modify these templates or even build your own. In this document, we will discuss how to do this.

The following is the standard way you may initialize the Model Card Toolkit.

    mct_directory = ...  # where the Model Card assets will be generated
    mct = ModelCardToolkit(mct_directory)
    model_card = mct.scaffold_assets()
    ... # set the model_card's fields here
    mct.update_model_card_json(model_card)

When you run `mct.scaffold_assets()`, the contents of [model_card_toolkit/template](https://github.com/tensorflow/model-card-toolkit/tree/master/model_card_toolkit/template) are copied into `mct_directory/template`. This includes premade templates such as [default_template.html.jinja](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/template/html/default_template.html.jinja) and [default_template.md.jinja](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/template/md/default_template.md.jinja).

The `model_card` object generated above can be enriched with fields from
[model_card.py](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card.py).
Once you are ready to generate a Model Card document, you can pass the
`model_card` back into MCT with `mct.update_model_card_json(model_card)`.

### Use a Premade Model Card Template

We can then generate a Model Card document using one of the default templates, via the code below.

    template_path = os.path.join(mct_directory, 'template/html/default_template.html.jinja')
    mct.export_format(template_pah, 'model_card.html')

### Modify the Model Card Template

You can freely modify a premade template to change styling, reorganize information, etc. You should be familiar with the [Jinja API](https://jinja.palletsprojects.com/en/2.11.x/api/) and [control structures](https://jinja.palletsprojects.com/en/2.11.x/templates/#list-of-control-structures). Model Card field names are taken from [model_card.py](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/model_card.py).

### Creating a new Model Card Template

Creating a new Model Card template works the same as modifying an existing one.

    my_custom_template_path = ...  # where the template is stored
    mct.export_format(my_custom_template_path, 'model_card.html')  # generate the final Model Card
