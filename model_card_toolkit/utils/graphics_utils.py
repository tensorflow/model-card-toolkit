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
"""Utilities for generating plots and graphics."""

import base64
import io

try:
  from matplotlib.figure import Figure
except:
  Figure = None


def figure_to_base64str(fig: Figure) -> str:
  """Converts a Matplotlib figure to a base64 string encoding.

  Args:
    fig: A matplotlib Figure.

  Returns:
    A base64 encoding of the figure.
  """
  buf = io.BytesIO()
  fig.savefig(buf, bbox_inches='tight', format='png')
  return base64.b64encode(buf.getbuffer().tobytes()).decode('ascii')
