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

  Attributes:
    name: The name of the version.
    date: The date this version was released.
    diff: The changes from the previous version.
  """
  name: Optional[Text] = None
  date: Optional[Text] = None
  diff: Optional[Text] = None


@dataclasses.dataclass
class Owner:
  """The information about owners of a model.

  Attributes:
    name: The name of the model owner.
    contact: The contact information for the model owner or owners. These
      could be individual email addresses, a team mailing list expressly, or a
      monitored feedback form.
  """
  name: Optional[Text] = None
  contact: Optional[Text] = None


@dataclasses.dataclass
class ModelDetails:
  """This section provides a general, high-level description of the model.

  Attributes:
    name: The name of the model.
    overview: A description of the model card.
    owners: The individuals or teams who own the model.
    version: The version of the model.
      license: The license information for the model. If the model is licensed
      for use by others, include the license type. If the model is not licensed
      for future use, you may state that here as well.
    references: Provide any additional links the reader may need. You can
      link to foundational research, technical documentation, or other materials
      that may be useful to your audience.
    citation: How should the model be cited? If the model is based on
      published academic research, cite the research.
  """
  name: Optional[Text] = None
  overview: Optional[Text] = None
  owners: List[Owner] = dataclasses.field(default_factory=list)
  version: Optional[Version] = dataclasses.field(default_factory=Version)
  license: Optional[Text] = None
  references: List[Text] = dataclasses.field(default_factory=list)
  citation: Optional[Text] = None


@dataclasses.dataclass
class Graphic:
  """A named inline plot.

  Attributes:
    name: The name of the graphic.
    image: The image string encoded as a base64 string.
  """
  name: Text
  image: Text


@dataclasses.dataclass
class Graphics:
  """A collection of graphics.

  Each ```graphic``` in the ```collection``` field has both a ```name``` and
  an ```image```. For instance, you might want to display a graph showing the
  number of examples belonging to each class in your training dataset:

  ```python

  model_card.model_parameters.data.train.graphics.collection = [
    {'name': 'Training Set Size', 'image': training_set_size_barchart},
  ]
  ```

  Then, provide a description of the graph:

  ```python

  model_card.model_parameters.data.train.graphics.description = (
    'This graph displays the number of examples belonging to each class ',
    'in the training dataset. ')
  ```

  Attributes:
    description: The name of the dataset.
    collection: A collection of graphics.
  """
  description: Optional[Text] = None
  collection: List[Graphic] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Dataset:
  """Provide some information about a dataset used to generate a model.

  Attributes:
    name: The name of the dataset.
    link: A link to the dataset.
    sensitive: Does this dataset contain human or other sensitive data?
    graphics: Visualizations of the dataset.
  """
  name: Optional[Text] = None
  link: Optional[Text] = None
  sensitive: Optional[bool] = None
  graphics: Graphics = dataclasses.field(default_factory=Graphics)


@dataclasses.dataclass
class Data:
  """The related datasets used to train and evaluate the model.

  Attributes:
    train: The training dataset.
    eval: The evaluation dataset.
  """
  train: Dataset = dataclasses.field(default_factory=Dataset)
  eval: Dataset = dataclasses.field(default_factory=Dataset)


@dataclasses.dataclass
class ModelParameters:
  """Parameters for construction of the model.

  Attributes:
    model_architecture: specifies the architecture of your model.
    data: specifies the datasets used to train and evaluate your model.
    input_format: describes the data format for inputs to your model.
    output_format: describes the data format for outputs from your model.
  """
  model_architecture: Optional[Text] = None
  data: Data = dataclasses.field(default_factory=Data)
  input_format: Optional[Text] = None
  output_format: Optional[Text] = None


@dataclasses.dataclass
class ConfidenceInterval:
  """The confidence interval of the metric.

  Attributes:
    lower_bound: The lower bound of the confidence interval.
    upper_bound: The upper bound of the confidence interval.
  """
  lower_bound: float
  upper_bound: float


@dataclasses.dataclass
class PerformanceMetric:
  """The details of the performance metric.

  Attributes:
    type: What performance metric are you reporting on?
    value: What is the value of this performance metric?
    confidence_interval: What is the confidence interval for this
      performance metric's value?
    threshold: At what threshold was this metric computed?
    slice: What slice of your data was this metric computed on?
  """
  type: Text
  value: Union[int, float, Text]
  confidence_interval: Optional[ConfidenceInterval] = None
  threshold: Optional[float] = None
  slice: Optional[Text] = None


@dataclasses.dataclass
class QuantitativeAnalysis:
  """The quantitative analysis of a model.

  Identify relevant performance metrics and display values. Let’s say you’re
  interested in displaying the accuracy and false positive rate (FPR) of a
  cat vs. dog classification model. Assuming you have already computed both
  metrics, both overall and per-class, you can specify metrics like so:

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

  Attributes:
    performance_metrics: The performance metrics being reported.
    graphics: A collection of visualizations of model performance.
  """
  performance_metrics: List[PerformanceMetric] = dataclasses.field(
      default_factory=list)
  graphics: Graphics = dataclasses.field(default_factory=Graphics)


@dataclasses.dataclass
class Risk:
  """Information about risks involved when using the model.

  Attributes:
    name: The name of the risk.
    mitigation_strategy: A mitigation strategy that you've implemented, or
      one that you suggest to users.
  """
  name: Text
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

  Attributes:
    users: Who are the intended users of the model? This may include
      researchers, developers, and/or clients. You might also include
      information about the downstream users you expect to interact with your
      model.
    use_cases: What are the intended use cases of the model? What use cases
      are out-of-scope?
    limitations: What are the known limitations of the model? This may
      include technical limitations, or conditions that may degrade model
      performance.
    tradeoffs: What are the known accuracy/performance tradeoffs for the
      model?
    ethical_considerations: What are the ethical risks involved in
      application of this model? For each risk, you may also provide a
      mitigation strategy that you've implemented, or one that you suggest to
      users.
  """
  users: List[Text] = dataclasses.field(default_factory=list)
  use_cases: List[Text] = dataclasses.field(default_factory=list)
  limitations: List[Text] = dataclasses.field(default_factory=list)
  tradeoffs: List[Text] = dataclasses.field(default_factory=list)
  ethical_considerations: List[Risk] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelCard:
  """Fields used to generate the Model Card.

  Attributes:
    schema_version: The Model Card JSON schema version.
    model_details: Descriptive metadata for the model.
    model_parameters: Technical metadata for the model.
    quantitative_analysis: Quantitative analysis of model performance.
    considerations: Any considerations related to model construction,
      training, and application.
  """
  schema_version: Optional[Text] = None
  model_details: ModelDetails = dataclasses.field(default=ModelDetails())
  model_parameters: ModelParameters = dataclasses.field(
      default_factory=ModelParameters)
  quantitative_analysis: QuantitativeAnalysis = dataclasses.field(
      default_factory=QuantitativeAnalysis)
  considerations: Considerations = dataclasses.field(
      default_factory=Considerations)

  def to_dict(self) -> Dict[Text, Any]:
    """Convert your model card to a python dictionary."""
    # ignore None properties recusively to allow missing values.
    ignore_none = lambda properties: {k: v for k, v in properties if v}
    return dataclasses.asdict(self, dict_factory=ignore_none)

  def to_json(self) -> Text:
    """Convert your model card to json."""
    return json.dumps(self.to_dict(), indent=2)
