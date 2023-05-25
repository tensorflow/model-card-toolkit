# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for rendering model cards."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
  from importlib.resources import files
except ImportError:
  from importlib_resources import files

import jinja2

from model_card_toolkit.utils import io_utils


def default_html_template() -> Path:
  """Returns the path to the default HTML template."""
  return files('model_card_toolkit'
               ).joinpath('template', 'html', 'default_template.html.jinja')


def default_md_template() -> Path:
  """Returns the path to the default Markdown template."""
  return files('model_card_toolkit'
               ).joinpath('template', 'md', 'default_template.md.jinja')


def render(
    template_path: Union[Path, str],
    output_path: Optional[Union[Path, str]] = None,
    template_variables: Optional[Dict[str, Any]] = None,
) -> str:
  """Renders a Jinja template and returns the content as a string.

  Args:
    template_path: The path to a Jinja template file.
    output_path: The path to write the rendered template to. If not provided,
      the rendered template will not be written to a file. If the file already
      exists, it will be overwritten.
    template_variables: A dictionary of variables to pass to the template.
  """
  template_variables = template_variables or {}
  template_dir = os.path.dirname(template_path)
  template_file = os.path.basename(template_path)
  jinja_env = jinja2.Environment(
      loader=jinja2.FileSystemLoader(template_dir),
      autoescape=True,
      auto_reload=True,
      cache_size=0,
  )

  template = jinja_env.get_template(template_file)
  content = template.render(template_variables)
  if output_path:
    io_utils.write_file(output_path, content)

  return content
