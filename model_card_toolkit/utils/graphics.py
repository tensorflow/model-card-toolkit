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
"""Utilities for generating model card plots/graphics."""

import base64
import io
import logging
from typing import Optional, Sequence, Tuple, Union

import attr
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_model_analysis as tfma
from tensorflow_metadata.proto.v0 import statistics_pb2

from model_card_toolkit import model_card as model_card_module

_COLOR_PALETTE = {
    'material_cyan_700': '#129EAF',  # default
    'material_teal_700': '#00897B',  # training dataset
    'material_indigo_400': '#5C6BC0',  # eval dataset
    'material_purple_500': '#A142F4'  # quantitative analysis
}


@attr.s(auto_attribs=True)
class _Graph():
  """Model Card graph."""

  # Necessary data to draw a graph.
  x: Optional[Sequence[Union[str, int, float]]] = None
  xerr: Optional[Sequence[Sequence[Union[int, float]]]] = None
  y: Optional[Sequence[Union[str, int, float]]] = None
  xlabel: Optional[str] = None
  ylabel: Optional[str] = None
  title: Optional[str] = None
  name: Optional[str] = None
  color: str = _COLOR_PALETTE['material_cyan_700']

  # Graph generated from the data above.
  figure: Optional[matplotlib.figure.Figure] = None
  base64str: Optional[str] = None


def annotate_dataset_feature_statistics_plots(
    model_card: model_card_module.ModelCard,
    data_stats: Sequence[statistics_pb2.DatasetFeatureStatisticsList]
) -> None:
  """Annotates visualizations for every dataset and feature.

  This function adds a new Dataset object at model_card.model_parameters.data
  for every dataset in data_stats. For every feature, histograms are created
  and encoded as base64 text strings. They can be found in the Dataset.graphics
  field.

  Args:
    model_card: The model card object.
    data_stats: A list of DatasetFeatureStatisticsList related to the dataset.
  """
  colors = (
      _COLOR_PALETTE['material_teal_700'],
      _COLOR_PALETTE['material_indigo_400']
  )
  for stats, color in zip(data_stats, colors):
    if not stats:
      continue
    for dataset in stats.datasets:
      graphs = []
      for feature in dataset.features:
        graph = _extract_graph_data_from_dataset_feature_statistics(
            feature, color
        )
        graph = _draw_histogram(graph)
        if graph is not None:
          graphs.append(
              model_card_module.Graphic(
                  name=graph.name, image=graph.base64str
              )
          )
      model_card.model_parameters.data.append(
          model_card_module.Dataset(
              name=dataset.name,
              graphics=model_card_module.GraphicsCollection(collection=graphs)
          )
      )


def annotate_eval_result_plots(
    model_card: model_card_module.ModelCard, eval_result: tfma.EvalResult
) -> None:
  """Annotates visualizations for every metric in eval_result.

  This function generates barcharts for sliced metrics, encoded as base64 text
  strings, and appends them to
  model_card.quantitative_analysis.graphics.collection.

  Args:
    model_card: The model card object.
    eval_result: A `tfma.EvalResult`.
  """

  # get all metric and slice names
  metrics = set()
  slices_keys = set()
  for slicing_metric in eval_result.slicing_metrics:
    slices_key, _ = stringify_slice_key(slicing_metric[0])
    if slices_key != 'Overall':
      slices_keys.add(slices_key)
    for output_name in slicing_metric[1]:
      for sub_key in slicing_metric[1][output_name]:
        metrics.update(slicing_metric[1][output_name][sub_key].keys())

  # generate barcharts based on metrics and slices
  graphs = []
  if not slices_keys:
    slices_keys.add('')
  for metric in metrics:
    for slices_key in slices_keys:
      graph = _extract_graph_data_from_slicing_metrics(
          eval_result.slicing_metrics, metric, slices_key
      )
      graph = _draw_histogram(graph)
      if graph is not None:
        graphs.append(graph)

  # annotate model_card with generated graphs
  model_card.quantitative_analysis.graphics.collection.extend(
      [
          model_card_module.Graphic(name=graph.name, image=graph.base64str)
          for graph in graphs
      ]
  )


def _extract_graph_data_from_dataset_feature_statistics(
    feature_stats: statistics_pb2.FeatureNameStatistics,
    color: Optional[str] = None
) -> Union[_Graph, None]:
  """Generates a _Graph object based on the histograms of feature_stats.

  Each bar in the histogram corresponds to a bucket in histogram.buckets.

  The bar heights are determined by the `sample_count` field of
  histogram.buckets. The bar labels are determined by the `label` field in a
  RankHistogram, or the bucket's `low_value` and `high_value` endpoints in a
  Histogram.

  Args:
    feature_stats: a FeatureNameStatistics proto.
    color: the colors of the barchart.

  Returns:
    _Graph or None if feature_stats is not num_stats or string_stats.
  """
  feature_name = feature_stats.name or feature_stats.path.step[0]
  graph = _Graph()

  if feature_stats.HasField(
      'num_stats'
  ) and feature_stats.num_stats.histograms:
    # Only generate graph for the first histogram.
    # The second one is QUANTILES graph.
    histogram = feature_stats.num_stats.histograms[0]
    graph.x = [int(bucket.sample_count) for bucket in histogram.buckets]
    graph.xlabel = 'counts'
    graph.y = [
        f'{bucket.low_value:.2f}-{bucket.high_value:.2f}'
        for bucket in histogram.buckets
    ]
    graph.ylabel = 'buckets'
    graph.title = f'counts | {feature_name}' if feature_name else 'counts'
    graph.name = f'counts | {feature_name}' if feature_name else 'counts'
    if color:
      graph.color = color
    return graph

  if feature_stats.HasField('string_stats'):
    rank_histogram = feature_stats.string_stats.rank_histogram
    graph.x = [int(bucket.sample_count) for bucket in rank_histogram.buckets]
    graph.xlabel = 'counts'
    graph.y = [bucket.label for bucket in rank_histogram.buckets]
    graph.ylabel = 'buckets'
    graph.title = f'counts | {feature_name}' if feature_name else 'counts'
    graph.name = f'counts | {feature_name}' if feature_name else 'counts'
    if color:
      graph.color = color
    return graph

  logging.warning(
      'Did not generate a graph for feature %s: '
      'FeatureNameStatistics must have string_stats or num_stats', feature_name
  )
  return None


def _extract_graph_data_from_slicing_metrics(
    slicing_metrics: Sequence[tfma.view.SlicedMetrics],
    metric: str,
    slices_key: str = '',
    output_name: str = '',
    sub_key: str = '',
) -> Optional[_Graph]:
  """Generates a barchart for a metric.

  Each bar in the barchart represents a slice in slicing_metrics. The size of
  a bar indicates the value of the metric for that slice.

  Args:
    slicing_metrics: A sequence of `tfma.view.SlicedMetrics` objects, where each
      `tfma.view.SlicedMetrics` corresponds to a different slice.
    metric: The name of a metric.
    slices_key: Only the slices with this slices_key will be added into graph.
      If it's an empty string, then all the the slices will be included.
    output_name: The output_name of interest in the slicing_metrics. '' by
      default.
    sub_key: The sub_key of interest in the slicing_metrics. '' by default.

  Returns:
    A Graph object, or None if any of the following are true:
      * metrics values format are not doubleValue or boundedValue.
      * the metric name ends with "_diff".
      * the metric name is "__ERROR__".
  """
  if metric.endswith('_diff') or metric == '__ERROR__':
    return None

  metric_values = []
  bounds = []
  slice_values = []
  has_bounded_value = False
  for slicing_metric in slicing_metrics:
    key, value = stringify_slice_key(slicing_metric[0])
    if key != 'Overall' and slices_key and key != slices_key:
      continue
    slice_values.append(value)

    if (
        output_name not in slicing_metric[1]
        or sub_key not in slicing_metric[1][output_name]
        or metric not in slicing_metric[1][output_name][sub_key]
    ):
      logging.warning(
          '%s, %s, %s not in %s. Skipping %s', output_name, sub_key, metric,
          slicing_metric[1], slices_key
      )
      return None

    # https://www.tensorflow.org/tfx/model_analysis/metrics#metric_value
    metric_value = slicing_metric[1][output_name][sub_key][metric]
    if 'doubleValue' in metric_value:
      metric_values.append(metric_value['doubleValue'])
      bounds.append((metric_value['doubleValue'], metric_value['doubleValue']))
    elif 'boundedValue' in metric_value:
      has_bounded_value = True
      metric_values.append(metric_value['boundedValue']['value'])
      bounds.append(
          (
              metric_value['boundedValue']['lowerBound'],
              metric_value['boundedValue']['upperBound']
          )
      )
    else:
      logging.warning(
          '%s must be a doubleValue or boundedValue; skipping %s.', metric,
          slices_key
      )
      return None

  graph = _Graph()
  graph.x = metric_values
  graph.y = slice_values
  if has_bounded_value:
    graph.xerr = [
        [
            float(metric_value) - float(bound[0])
            for metric_value, bound in zip(metric_values, bounds)
        ],
        [
            float(bound[1]) - float(metric_value)
            for metric_value, bound in zip(metric_values, bounds)
        ],
    ]
  graph.xlabel = metric
  graph.ylabel = 'slices'
  graph.name = f'{metric} | {slices_key}' if slices_key else metric
  graph.title = f'{metric} | {slices_key}' if slices_key else metric
  graph.color = _COLOR_PALETTE['material_purple_500']
  return graph


def _draw_histogram(graph: _Graph) -> Optional[_Graph]:
  """Draw a histogram given the graph.

  Args:
    graph: The _Graph object represents the necessary data to draw a histogram.

  Returns:
    A _Graph object, or None if plotting raises TypeError given the raw data.
  """
  if not graph:
    return None
  try:
    # generate and open a new figure
    figure, ax = plt.subplots()
    # When graph.x or y is str, the histogram is ill-defined.
    ax.barh(graph.y, graph.x, color=graph.color)
    ax.set_title(graph.title)
    if graph.xlabel:
      ax.set_xlabel(graph.xlabel)
    if graph.ylabel:
      ax.set_ylabel(graph.ylabel)
    for index, value in enumerate(graph.x):
      show_value = f'{value:.2f}' if isinstance(value, float) else value
      # To avoid the number has overlap with the box of the graph.
      if value > 0.9 * max(graph.x):
        ax.text(
            value - (value / 10), index, show_value, va='center', color='w'
        )
      else:
        ax.text(value, index, show_value, va='center')

    graph.figure = figure
    graph.base64str = figure_to_base64str(figure)
  except TypeError as e:
    logging.info('skipping %s for histogram; plot error: %s:', graph.name, e)
    return None
  finally:
    # closes the figure (to limit memory consumption)
    plt.close()
  return graph


def figure_to_base64str(fig: matplotlib.figure.Figure) -> str:
  """Converts a Matplotlib figure to a base64 string encoding.

  Args:
    fig: A matplotlib Figure.

  Returns:
    A base64 encoding of the figure.
  """
  buf = io.BytesIO()
  fig.savefig(buf, bbox_inches='tight', format='png')
  return base64.b64encode(buf.getbuffer().tobytes()).decode('ascii')


# FeatureValueType represents a value that a feature could take.
FeatureValueType = Union[str, int, float]  # pylint: disable=invalid-name

# SingletonSliceKeyType is a tuple, where the first element is the key of the
# feature, and the second element is its value. This describes a single
# feature-value pair.
SingletonSliceKeyType = Tuple[str, FeatureValueType]  # pylint: disable=invalid-name

# SliceKeyType is a either the empty tuple (for the overal slice) or a tuple of
# SingletonSliceKeyType. This completely describes a single slice.
SliceKeyType = Union[Tuple[()], Tuple[SingletonSliceKeyType, ...]]  # pylint: disable=invalid-name


def stringify_slice_key(slice_key: SliceKeyType) -> Tuple[str, str]:
  """Stringifies a slice key.

  The string representation of a SingletonSliceKeyType is "feature:value". When
  multiple columns / features are specified, the string representation of a
  SliceKeyType is "c1, c2, ...:v1, v2, ..." where c1, c2, ... are the column
  names and v1, v2, ... are the corresponding values For example,
  ('gender, 'f'), ('age', 5) befores age, gender:f, 5. If no columns / feature
  specified, return "Overall".

  Note that we do not perform special escaping for slice values that contain
  ', '. This stringified representation is meant to be human-readbale rather
  than a reversible encoding.

  The columns will be in the same order as in SliceKeyType. If they are
  generated using SingleSliceSpec.generate_slices, they will be in sorted order,
  ascending.

  Technically float values are not supported, but we don't check for them here.

  Args:
    slice_key: Slice key to stringify. The constituent SingletonSliceKeyTypes
      should be sorted in ascending order.

  Returns:
    A tuple of string representation of the slice key and slice value.
  """
  key_count = len(slice_key)
  if not key_count:
    return ('Overall', 'Overall')

  keys = []
  values = []
  separator = ', '

  for (feature, value) in slice_key:
    keys.append(feature)
    values.append(value)

  # To use u'{}' instead of '{}' here to avoid encoding a unicode character with
  # ascii codec.
  return (
      separator.join([u'{}'.format(key) for key in keys]),
      separator.join([u'{}'.format(value) for value in values])
  )
