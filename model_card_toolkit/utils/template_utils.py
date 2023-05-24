import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
  from importlib.resources import files
except ImportError:
  from importlib_resources import files

import jinja2

from model_card_toolkit.utils import io_utils

def default_html_template() -> str:
  return str(files('model_card_toolkit').joinpath(
      'template', 'html', 'default_template.html.jinja'
  ))


def default_md_template() -> str:
  return str(files('model_card_toolkit').joinpath(
      'template', 'md', 'default_template.md.jinja'
  ))

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
