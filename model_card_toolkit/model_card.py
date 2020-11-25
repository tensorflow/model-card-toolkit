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
  """The information about verions of a model."""
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
  # The contact information of the owner.
  contact: Optional[Text] = None


@dataclasses.dataclass
class ModelDetails:
  """Metadata about the model."""
  # The name of the model.
  name: Optional[Text] = None
  # A description of the model card.
  overview: Optional[Text] = None
  # The individuals or teams who own the model.
  owners: List[Owner] = dataclasses.field(default_factory=list)
  # The version of the model.
  version: Version = dataclasses.field(default_factory=Version)
  # The model's license for use.
  license: Optional[Text] = None
  # Links providing more information about the model.
  references: List[Text] = dataclasses.field(default_factory=list)
  # How to reference this model card.
  citation: Optional[Text] = None


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
  """The information about a dataset used to generate a model."""
  # The name of the dataset.
  name: Optional[Text] = None
  # The contact information of the owner
  link: Optional[Text] = None
  # Does this dataset contain human or other sensitive data?
  sensitive: Optional[bool] = None
  # Visualizations of the dataset.
  graphics: Graphics = dataclasses.field(default_factory=Graphics)


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
  # The datasets used to train and evaluate the model.
  data: Data = dataclasses.field(default_factory=Data)
  # The data format for inputs to the model.
  input_format: Optional[Text] = None
  # The data format for outputs from the model.
  output_format: Optional[Text] = None


@dataclasses.dataclass
class ConfidenceInterval:
  """The confidence interval of the metric."""
  # The lower bound of the confidence interval.
  lower_bound: float
  # The upper bound of the confidence interval.
  upper_bound: float


@dataclasses.dataclass
class PerformanceMetric:
  """The details of the performance metric."""
  # The type of performance metric.
  type: Text
  # The value of the performance metric.
  value: Union[int, float, Text]
  # The confidence interval of the metric.
  confidence_interval: Optional[ConfidenceInterval] = None
  # The decision threshold the metric was computed on.
  threshold: Optional[float] = None
  # The name of the slice this metric was computed on.
  slice: Optional[Text] = None


@dataclasses.dataclass
class QuantitativeAnalysis:
  """The quantitative analysis of a model."""
  # The model performance metrics being reported.
  performance_metrics: List[PerformanceMetric] = dataclasses.field(
      default_factory=list)
  # Visualizations of model performance.
  graphics: Graphics = dataclasses.field(default_factory=Graphics)


@dataclasses.dataclass
class Risk:
  """The information about risks when using the model."""
  # The name of the risk.
  name: Text
  # Strategy used to address this risk.
  mitigation_strategy: Text


@dataclasses.dataclass
class Considerations:
  """Considerations related to model construction, training, and application."""
  # Who are the intended users of the model?
  users: List[Text] = dataclasses.field(default_factory=list)
  # What are the intended use cases of the model.
  use_cases: List[Text] = dataclasses.field(default_factory=list)
  # What are the known technical limitations of the model.
  limitations: List[Text] = dataclasses.field(default_factory=list)
  # What are the known tradeoffs in accuracy/performance of the model
  tradeoffs: List[Text] = dataclasses.field(default_factory=list)
  # What are the ethical risks involved in the application of this model.
  ethical_considerations: List[Risk] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelCard:
  """Fields used to generate the Model Card."""
  # The json schema version of the ModelCard
  schema_version: Optional[Text] = None
  # Descriptive metadata for the model.
  model_details: ModelDetails = dataclasses.field(default=ModelDetails())
  # Parameters used when generating the model.
  model_parameters: ModelParameters = dataclasses.field(
      default_factory=ModelParameters)
  # The quantitative analysis of the ModelCard
  quantitative_analysis: QuantitativeAnalysis = dataclasses.field(
      default_factory=QuantitativeAnalysis)
  # The considerations related to model construction, training, and application.
  considerations: Considerations = dataclasses.field(
      default_factory=Considerations)

  def to_dict(self) -> Dict[Text, Any]:
    # ignore None properties recusively to allow missing values.
    ignore_none = lambda properties: {k: v for k, v in properties if v}
    return dataclasses.asdict(self, dict_factory=ignore_none)

  def to_json(self) -> Text:
    return json.dumps(self.to_dict(), indent=2)

  def from_json(self, json_str: Text) -> None:
    model_card_dict = json.loads(json_str)
    self.schema_version = model_card_dict.get('schema_version')
    self.model_details = ModelDetails(**(model_card_dict.get('model_details')))
    self.model_parameters = ModelParameters(
        **(model_card_dict.get('model_parameters')))
    self.quantitative_analysis = QuantitativeAnalysis(
        **(model_card_dict.get('quantitative_analysis')))
    self.considerations = Considerations(
        **(model_card_dict.get('considerations')))
