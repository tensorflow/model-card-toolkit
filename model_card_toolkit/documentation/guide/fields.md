# Model Card Fields

In this document, we'll provide a brief explanation of each of the fields
included in the Model Card Toolkit JSON schema. You can use this as a reference
as you're creating your Model Card.

[TOC]

## Model Details

This section provides a general, high-level description of the model. The
details in this section should be fairly straightforward. They are as follows:

##### `model_card.model_details.name`

Provide the name of the model.

##### `model_card.model_details.overview`

This is a general description of the model.

##### `model_card.model_details.version`

This field refers to the model version name. If there are multiple versions of
the model, or there may be in the future, it’s useful for your audience to know
which version of the model is discussed in the Model Card. If there are previous
versions of this model, briefly describe how this version is different. If there
will only ever be one version of the model released, this field may be omitted.

##### `model_card.model_details.date`

When was the model released?

##### `model_card.model_details.owners.name`

The “owners” field provides your audience with some accountability and
transparency into who was involved in the models’ creation. You can list all of
the model contributors individually, or the team responsible for the model.

##### `model_card.model_details.owners.contact`

If possible, it’s helpful to provide some contact information for readers who
may have further questions. This could be individual email addresses, a team
mailing list expressly created for this purpose, or a monitored feedback form.

##### `model_card.model_details.license`

If the model is licensed for use by others, include the license type. If the
model is not licensed for future use, you may state that here as well.

##### `model_card.model_details.citation`

How should the model be cited? If the model is based on published academic
research, cite the research.

##### `model_card.model_details.references`

Provide any additional information the reader may need. You can link to
foundational research, technical documentation, or other materials that may be
useful to your audience.

## Model Parameters

In the Model Parameters section, you'll include more technical details about
your model:

##### `model_card.model_parameters.model_architecture`

Specify the model architecture for your model.

##### `model_card.model_parameters.data`

Specify the datasets used to train and evaluate your model.

##### `model_card.model_parameters.data.{train|eval}.graphics`

You can also display graphics for your dataset using
`model_card.model_parameters.data.eval.graphics.collection` or
`model_card.model_parameters.data.train.graphics.collection` by passing in an
array of graphics where each graphic has both a name and an image. You can do
this for your training set and your evaluation set.

For instance, you might want to display a graph showing the number of examples
belonging to each class in your training dataset:

```python
model_card.model_parameters.data.train.graphics.collection = [
  {'name': 'Training Set Size', 'image': training_set_size_barchart},
]
```

Then provide a description of the graph:

```python
model_card.model_parameters.data.train.graphics.description = (
  'This is a graph displaying the number of examples belonging to each class '
  'in our training dataset. '
)
```

This is particularly important if you’ve used a class-imbalanced dataset to
train your model, since this could have performance implications.

##### `model_card.model_parameters.input_format`

Describe the data format for inputs to your model.

##### `model_card.model_parameters.output_format`

Describe the data format for outputs from your model.

## Considerations

The considerations section includes qualitative information about your model,
including some analysis of its risks and limitations. As such, this section
usually requires careful consideration, and conversations with many relevant
stakeholders, including other model developers, dataset producers, and
downstream users likely to interact with your model, or be affected by its
outputs.

##### `model_card.considerations.use_cases`

Start with an accessible, but technically precise description of what the model
does, and the domains in which you intend it to be used. You may include
**out-of-scope** applications in this section as well. Out-of-scope applications
are *not* applications that are beyond the model’s technical capabilities (which
you’ll report on in the “limitations” section), but use cases in which you do
not want your model applied.

##### `model_card.considerations.limitations`

This section refers to the model’s limitations, *within* its intended usage. In
this section, you can report on technical limitations, and input types, or
environmental conditions that might degrade model performance.

##### `model_card.considerations.ethical_considerations`

In this section, you should report on any ethical considerations or risks you’ve
identified. For each risk, you can also provide a mitigation strategy you’ve
implemented, or one you suggest to users. For instance:

```python
model_card.considerations.ethical_considerations = [Risk(name='If the cat vs dog classifier is run on a photo of a human, it will '
        'return “cat” or “dog,” which may be offensive. ', mitigation_strategy='When users upload a photo in the UI, a dialog box will prompt them to '
        'confirm that the photo is either a cat or dog.')]]
```

## Quantitative Analysis

In this section, you’ll display quantitative analysis of your model. If your
model performs differently across different labels or attributes, it’s important
to report on this here.

##### `model_card.quantitative_analysis.performance_metrics`

Identify relevant performance metrics and display values. Let’s say you’re
interested in displaying the *accuracy* and *false positive rate (FPR)* of your
cat vs. dog classification model. Assuming you have already computed both
metrics, both overall and per-class, you can specify metrics:

```python
model_card.quantitative_analysis.performance_metrics = [
  {'type': 'accuracy', 'value': computed_accuracy},
  {'type': 'accuracy', 'value': cat_accuracy, 'slice': 'cat'},
  {'type': 'accuracy', 'value': dog_accuracy, 'slice': 'dog'},
  {'type': 'fpr', 'value': computed_fpr},
  {'type': 'fpr', 'value': cat_fpr, 'slice': 'cat'},
  {'type': 'fpr', 'value': dog_fpr, 'slice': 'dog'},
]
```

##### `model_card.quantitative_analysis.performance_metrics.confidence_interval`

Optionally, you can supply a `confidence_interval` for each metric, specifying
the `lower_bound` and `upper_bound`.

##### `model_card.quantitative_analysis.performance_metrics.threshold`

If the metric was computed at a given threshold, specify that in the `threshold`
property.

##### `model_card.quantitative_analysis.graphics`

You can also include graphs of model performance in the same manner as for your
dataset. To display a bar chart illustrating the accuracy of your classifier:

```python
model_card.quantitative_analysis.graphics.collection = [
  {'name': 'Classifier Accuracy', 'image': accuracy_barchart},
]
```

Then provide a description for the graphics in the collection:

```python
model_card.quantitative_analysis.graphics.description = (
  'Graphs for our cat vs. dog classifier.')
```
