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

import attr


@attr.s(auto_attribs=True)
class Version():
  """The information about verions of a model."""
  # The name of the version.
  name: Optional[Text] = None
  # The date the version was released.
  date: Optional[Text] = None
  # The changes from the previous version.
  diff: Optional[Text] = None


@attr.s(auto_attribs=True)
class Owner():
  """The information about owners of a model."""
  # The name of the owner.
  name: Optional[Text] = None
  # The contact information of the owner.
  contact: Optional[Text] = None


@attr.s(auto_attribs=True)
class ModelDetails():
  """Metadata about the model."""
  # The name of the model.
  name: Optional[Text] = None
  # A description of the model card.
  overview: Optional[Text] = None
  # The individuals or teams who own the model.
  owners: List[Owner] = attr.Factory(list)
  # The version of the model.
  version: Version = attr.Factory(Version)
  # The model's license for use.
  license: Optional[Text] = None
  # Links providing more information about the model.
  references: List[Text] = attr.Factory(list)
  # How to reference this model card.
  citation: Optional[Text] = None


@attr.s(auto_attribs=True)
class Graphic():
  """A named inline plot."""
  # The name of the graphic.
  name: Text
  # The image string encoded as a base64 string.
  image: Text


@attr.s(auto_attribs=True)
class Graphics():
  """A collection of graphics."""
  # A description of this collection of graphics.
  description: Optional[Text] = None
  # A collection of graphics.
  collection: List[Graphic] = attr.Factory(list)


@attr.s(auto_attribs=True)
class Dataset():
  """The information about a dataset used to generate a model."""
  # The name of the dataset.
  name: Optional[Text] = None
  # The contact information of the owner
  link: Optional[Text] = None
  # Does this dataset contain human or other sensitive data?
  sensitive: Optional[bool] = None
  # Visualizations of the dataset.
  graphics: Graphics = attr.Factory(Graphics)


@attr.s(auto_attribs=True)
class Data():
  """The related datasets used to train and evaluate the model."""
  # The training dataset
  train: Dataset = attr.Factory(Dataset)
  # The evaluation dataset
  eval: Dataset = attr.Factory(Dataset)


@attr.s(auto_attribs=True)
class ModelParameters():
  """Parameters for construction of the model."""
  # The architecture of the model.
  model_architecture: Optional[Text] = None
  # The datasets used to train and evaluate the model.
  data: Data = attr.Factory(Data)


@attr.s(auto_attribs=True)
class ConfidenceInterval():
  """The confidence interval of the metric."""
  # The lower bound of the confidence interval.
  lower_bound: float
  # The upper bound of the confidence interval.
  upper_bound: float


@attr.s(auto_attribs=True)
class PerformanceMetric():
  """The details of the performance metric."""
  # The type of performance metric.
  type: Text
  # The value of the performance metric.
  value: Union[int, float, Text]
  # The confidence interval of the metric.
  confidence_interval: ConfidenceInterval = attr.Factory(ConfidenceInterval)
  # The decision threshold the metric was computed on.
  threshold: Optional[float] = None
  # The name of the slice this metric was computed on.
  slice: Optional[Text] = None


@attr.s(auto_attribs=True)
class QuantitativeAnalysis():
  """The quantitative analysis of a model."""
  # The model performance metrics being reported.
  performance_metrics: List[PerformanceMetric] = attr.Factory(list)
  # Visualizations of model performance.
  graphics: Graphics = attr.Factory(Graphics)


@attr.s(auto_attribs=True)
class Risk():
  """The information about risks when using the model."""
  # The name of the risk.
  name: Text
  # Strategy used to address this risk.
  mitigation_strategy: Text


@attr.s(auto_attribs=True)
class Considerations():
  """Considerations related to model construction, training, and application."""
  # Who are the intended users of the model?
  users: List[Text] = attr.Factory(list)
  # What are the intended use cases of the model.
  use_cases: List[Text] = attr.Factory(list)
  # What are the known technical limitations of the model.
  limitations: List[Text] = attr.Factory(list)
  # What are the known tradeoffs in accuracy/performance of the model
  tradeoffs: List[Text] = attr.Factory(list)
  # What are the ethical risks involved in the application of this model.
  ethical_considerations: List[Risk] = attr.Factory(list)


@attr.s(auto_attribs=True)
class ModelCard(object):
  """A class that represents assets of ModelCards created by the MCT."""
  # The json schema version of the ModelCard
  schema_version: Optional[Text] = None
  # Descriptive metadata for the model.
  model_details: ModelDetails = attr.Factory(ModelDetails)
  # Parameters used when generating the model.
  model_parameters: ModelParameters = attr.Factory(ModelParameters)
  # The quantitative analysis of the ModelCard
  quantitative_analysis: QuantitativeAnalysis = attr.Factory(
      QuantitativeAnalysis)
  # The considerations related to model construction, training, and application.
  considerations: Considerations = attr.Factory(Considerations)

  def to_dict(self) -> Dict[Text, Any]:
    # ignore None properties recusively to allow missing values.
    return attr.asdict(self, filter=lambda attr, value: value)

  def to_json(self) -> Text:
    return json.dumps(self.to_dict(), indent=2)
