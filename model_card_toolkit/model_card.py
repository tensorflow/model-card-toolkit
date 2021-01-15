# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model Card Data Class.

The Model Card (MC) is the document designed for transparent reporting of AI
model provenance, usage, and ethics-informed evaluation. The model card can be
presented by different formats (e.g. HTML, PDF, Markdown). The properties of
the Model Card (MC) are defined by a json schema. The ModelCard class in the
ModelCardsToolkit serves as an API to read and write MC properties by the users.
"""

import json
from typing import Any, Dict, List, Optional, Text, Union

import dataclasses


@dataclasses.dataclass
class Version:
  """The information about verions of a model.

  If there are multiple versions of the model, or there may be in the future,
  it’s useful for your audience to know which version of the model is discussed
  in the Model Card. If there are previous versions of this model, briefly
  describe how this version is different. If no more than one version of the
  model will be released, this field may be omitted.
  """
  # The name of the version.
  name: Optional[Text] = None
  # The date the version was released.
  date: Optional[Text] = None
  # The changes from the previous version.
  diff: Optional[Text] = None


@dataclasses.dataclass
class Owner:
  """The information about owners of a model."""
  # The name of the owner.
  name: Optional[Text] = None
  """The name of the owner or owners.

  The “owners” field provides your audience with some accountability and
  transparency into who was involved in the models’ creation. You can list all
  of the model contributors individually, or the team responsible for the
  model.
  """
  # The contact information of the owner.
  contact: Optional[Text] = None
  """The contact information for the owner or owners.

  If possible, it’s helpful to provide some contact information for readers who
  may have further questions. This could be individual email addresses, a team
  mailing list expressly created for this purpose, or a monitored feedback form.
  """


@dataclasses.dataclass
class ModelDetails:
  """This section provides a general, high-level description of the model."""
  # The name of the model.
  name: Optional[Text] = None
  """The name of the model."""
  # A description of the model card.
  overview: Optional[Text] = None
  """Provide a general description of the model."""
  # The individuals or teams who own the model.
  owners: List[Owner] = dataclasses.field(default_factory=list)
  # The version of the model.
  version: Version = dataclasses.field(default_factory=Version)
  """The version of the model.

  If there are previous versions of this model, briefly describe how this
  version is different. If there will only ever be one version of the model
  released, you may omit ModelDetails.version."""
  # The model's license for use.
  license: Optional[Text] = None
  """The license information for the model.

  If the model is licensed for use by others, include the license type.
  If the model is not licensed for future use, you may state that here as well.
  """
  # Links providing more information about the model.
  references: List[Text] = dataclasses.field(default_factory=list)
  """Provide any additional information the reader may need.

  You can link to foundational research, technical documentation, or other
  materials that may be useful to your audience.
  """
  # How to reference this model card.
  citation: Optional[Text] = None
  """How should the model be cited?

  If the model is based on published academic research, cite the research.
  """


@dataclasses.dataclass
class Graphic:
  """A named inline plot."""
  # The name of the graphic.
  name: Text
  # The image string encoded as a base64 string.
  image: Text


@dataclasses.dataclass
class Graphics:
  """A collection of graphics."""
  # A description of this collection of graphics.
  description: Optional[Text] = None
  # A collection of graphics.
  collection: List[Graphic] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Dataset:
  """Provide some information about a dataset used to generate a model."""
  # The name of the dataset.
  name: Optional[Text] = None
  """The name of the dataset."""
  # The contact information of the owner.
  link: Optional[Text] = None
  """The contact information of the owner."""
  # Does this dataset contain human or other sensitive data?
  sensitive: Optional[bool] = None
  """Does this dataset contain human or other sensitive data?"""
  # Visualizations of the dataset.
  graphics: Graphics = dataclasses.field(default_factory=Graphics)
  """# Visualizations of the dataset."""


@dataclasses.dataclass
class Data:
  """The related datasets used to train and evaluate the model."""
  # The training dataset
  train: Dataset = dataclasses.field(default_factory=Dataset)
  # The evaluation dataset
  eval: Dataset = dataclasses.field(default_factory=Dataset)


@dataclasses.dataclass
class ModelParameters:
  """Parameters for construction of the model."""
  # The architecture of the model.
  model_architecture: Optional[Text] = None
  """Specify the model architecture for your model."""
  # The datasets used to train and evaluate the model.
  data: Data = dataclasses.field(default_factory=Data)
  """Specify the datasets used to train and evaluate your model."""
  # The data format for inputs to the model.
  input_format: Optional[Text] = None
  """Describe the data format for inputs to your model."""
  # The data format for outputs from the model.
  output_format: Optional[Text] = None
  """Describe the data format for outputs from your model."""


@dataclasses.dataclass
class ConfidenceInterval:
  """The confidence interval of the metric."""
  # The lower bound of the confidence interval.
  lower_bound: float
  """The lower bound of the confidence interval."""
  # The upper bound of the confidence interval.
  upper_bound: float
  """The upper bound of the confidence interval."""


@dataclasses.dataclass
class PerformanceMetric:
  """The details of the performance metric."""
  # The type of performance metric.
  type: Text
  """What performance metric are you reporting on?"""
  # The value of the performance metric.
  value: Union[int, float, Text]
  """What is the value of the performance metric?"""
  # The confidence interval of the metric.
  confidence_interval: Optional[ConfidenceInterval] = None
  # The decision threshold the metric was computed on.
  threshold: Optional[float] = None
  """At what threshold was this metric computed?"""
  # The name of the slice this metric was computed on.
  slice: Optional[Text] = None
  """What slice of your data was this metric computed on?

  Reporting on metrics for different slices of your data is highly relevant for
  determining possible fairness concerns.
  """


@dataclasses.dataclass
class QuantitativeAnalysis:
  """The quantitative analysis of a model.

  Identify relevant performance metrics and display values. Let’s say you’re
  interested in displaying the accuracy and false positive rate (FPR) of your
  cat vs. dog classification model. Assuming you have already computed both
  metrics, both overall and per-class, you can specify metrics:


  """
  # The model performance metrics being reported.
  performance_metrics: List[PerformanceMetric] = dataclasses.field(
      default_factory=list)
  # Visualizations of model performance.
  graphics: Graphics = dataclasses.field(default_factory=Graphics)

  """Display graphics for your model.

  You can also display graphics for your dataset using
  ```model_card.model_parameters.data.eval.graphics.collection``` or
  ```model_card.model_parameters.data.train.graphics.collection``` by passing in
  an array of ```graphic```s where each ```graphic``` has both a ```name``` and
  an ```image```. You can do this for your training set and your evaluation set.

  For instance, you might want to display a graph showing the number of examples
  belonging to each class in your training dataset:

  ```python

  model_card.model_parameters.data.train.graphics.collection = [
  {'name': 'Training Set Size', 'image': training_set_size_barchart},
  ]
  ```
  Then, provide a description of the graph:

  ```python

  model_card.model_parameters.data.train.graphics.description = (
  'This is a graph displaying the number of examples belonging to each class '
  'in our training dataset. '
  ```
  This is particularly important if you’ve used a class-imbalanced dataset to
  train your model, since this could have performance implications.
  """


@dataclasses.dataclass
class Risk:
  """The information about risks when using the model."""
  # The name of the risk.
  name: Text
  # Strategy used to address this risk.
  mitigation_strategy: Text


@dataclasses.dataclass
class Considerations:
  """Considerations related to model construction, training, and application.

  The considerations section includes qualitative information about your model,
  including some analysis of its risks and limitations. As such, this section
  usually requires careful consideration, and conversations with many relevant
  stakeholders, including other model developers, dataset producers, and
  downstream users likely to interact with your model, or be affected by its
  outputs.
  """
  # Who are the intended users of the model?
  users: List[Text] = dataclasses.field(default_factory=list)
  """Who are the intended users of the model?

  This may include researchers, developers, and/or clients. You might also
  include information about the downstream users you expect to interact with
  your model."""
  # What are the intended use cases of the model.
  use_cases: List[Text] = dataclasses.field(default_factory=list)
  """Provide an accessible, but technically precise description of what the
  model does

  You may also report on the domains in which it may be used. It's useful to
  describe any out-of-scope applications in this section as well. Out-of-scope
  applications are not applications that are beyond the model’s technical
  capabilities (which you’ll report on in the “limitations” section), but use
  cases in which you do not want your model applied.
  """

  # What are the known technical limitations of the model.
  limitations: List[Text] = dataclasses.field(default_factory=list)
  """This section refers to the model’s limitations, within its intended usage.

  In this section, you can report on technical limitations, and input types, or
  environmental conditions that might degrade model performance.
  """
  # What are the known tradeoffs in accuracy/performance of the model
  tradeoffs: List[Text] = dataclasses.field(default_factory=list)
  """Are there known tradeoffs in accuracy/performance for the model?"""
  # What are the ethical risks involved in the application of this model.
  ethical_considerations: List[Risk] = dataclasses.field(default_factory=list)
  """In this section, you should report on any ethical considerations or risks
  you’ve identified.

  For each risk, you can also provide a mitigation strategy
  you’ve implemented, or one you suggest to users. For instance:

  ```python

  model_card.considerations.ethical_considerations = [(
  'name':
      'If the cat vs dog classifier is run on a photo of a human, it will '
      'return “cat” or “dog,” which may be offensive.',
  'mitigation_strategy':
      'When users upload a photo in the UI, a dialog box will prompt them to '
      'confirm that the photo is either a cat or dog.')]]
  """


@dataclasses.dataclass
class ModelCard:
  """Fields used to generate the Model Card."""
  # The json schema version of the ModelCard
  schema_version: Optional[Text] = None
  """What version of the Model Card JSON schema are you using?"""
  # Descriptive metadata for the model.
  model_details: ModelDetails = dataclasses.field(default=ModelDetails())
  """Specify the descriptive metadata for the model."""
  # Parameters used when generating the model.
  model_parameters: ModelParameters = dataclasses.field(
      default_factory=ModelParameters)
  """Explain the technical details of the model."""
  # The quantitative analysis of the ModelCard
  quantitative_analysis: QuantitativeAnalysis = dataclasses.field(
      default_factory=QuantitativeAnalysis)
  """Provide quantitative analysis of your model's performance."""
  # The considerations related to model construction, training, and application.
  considerations: Considerations = dataclasses.field(
      default_factory=Considerations)
  """Report on ethical considerations you explored."""

  def to_dict(self) -> Dict[Text, Any]:
    """Convert your model card to a python dictionary."""
    # ignore None properties recusively to allow missing values.
    ignore_none = lambda properties: {k: v for k, v in properties if v}
    return dataclasses.asdict(self, dict_factory=ignore_none)

  def to_json(self) -> Text:
    """Convert your model card to json."""
    return json.dumps(self.to_dict(), indent=2)
