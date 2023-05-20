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

_COLOR_PALETTE = {
    'material_cyan_700': '#129EAF',  # default
    'material_teal_700': '#00897B',  # training dataset
    'material_indigo_400': '#5C6BC0',  # eval dataset
    'material_purple_500': '#A142F4'  # quantitative analysis
}


@attr.s(auto_attribs=True)
class Graph():
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


def draw_histogram(graph: Graph) -> Optional[Graph]:
  """Draw a histogram given the graph.

  Args:
    graph: The Graph object represents the necessary data to draw a histogram.

  Returns:
    A Graph object, or None if plotting raises TypeError given the raw data.
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
