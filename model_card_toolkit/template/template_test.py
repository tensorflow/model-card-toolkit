# Lint as: python3
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
"""Tests model_card_toolkit.template."""

import os
import re
from typing import List, Optional
from absl.testing import absltest
import attr
import jinja2
from model_card_toolkit import model_card

_GRAPHICS = model_card.Graphics(
    description='These are my graphics.',
    collection=[{
        'name': 'graphic',
        'image': 'I am an image beep boop'
    }])

_TRAIN_DATA = model_card.Dataset(name='train_split', link='path/to/train')
_TRAIN_DATA_WITH_GRAPHIC = model_card.Dataset(
    name='train_split', link='path/to/train', graphics=_GRAPHICS)
_EVAL_DATA = model_card.Dataset(name='eval_split', link='path/to/eval')
_EVAL_DATA_WITH_GRAPHIC = model_card.Dataset(
    name='eval_split', link='path/to/eval', graphics=_GRAPHICS)

_TEMPLATES_ROOT = os.path.dirname(__file__)
_FOLDER = 'html'
_TEMPLATE_FILE = 'default_template.html.jinja'

_MODEL_DETAILS_PROPERTIES = [
    'overview', 'owners', 'version', 'license', 'references', 'citation'
]


@attr.s(auto_attribs=True)
class ExpectedGraphic():
  """The expected graphics in a model card."""
  # Training dataset graphics.
  train: Optional[bool] = False
  # Eval dataset graphics.
  eval: Optional[bool] = False
  # Eval result graphics.
  quantitative: Optional[bool] = False

  def yes(self):
    return self.train or self.eval or self.quantitative


class TemplateTest(absltest.TestCase):

  def find(self, html: str, start_tag: str, end_tag: str) -> str:
    if start_tag in html:
      return html.split(start_tag)[1].split(end_tag)[0].strip()
    else:
      return ''

  def find_all(self, html: str, start_tag: str, end_tag: str) -> List[str]:
    start_tag_indices = [tag.end() for tag in re.finditer(start_tag, html)]
    end_tag_indices = [tag.start() for tag in re.finditer(end_tag, html)]
    self.assertEqual(
        len(start_tag_indices), len(end_tag_indices),
        (f'There are {len(start_tag_indices)} instances of {start_tag}, '
         f'but {len(end_tag_indices)} instances of {end_tag}.'))
    return [
        html[start:end].strip()
        for start, end in zip(start_tag_indices, end_tag_indices)
    ]

  def parse_ul(self, html: str) -> List[str]:
    ul = self.find(html, '<ul>', '</ul>')
    return self.find_all(ul, '<li>', '</li>')

  def assertRenderedTemplate(self, mc: model_card.ModelCard,
                             expected_graphic: ExpectedGraphic):
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(_TEMPLATES_ROOT, _FOLDER)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
        auto_reload=True,
        cache_size=0)
    page = jinja_env.get_template(_TEMPLATE_FILE).render(model_card=mc)

    with self.subTest(name='title'):
      title = self.find(page, '<title>', '</title>')
      self.assertEqual(title, 'Model Card for ' + mc.model_details['name'])

    with self.subTest(name='h1'):
      header1 = self.find(page, '<h1>', '</h1>')
      self.assertEqual(header1, 'Model Card for ' + mc.model_details['name'])

    with self.subTest(name='h2'):
      header2s = self.find_all(page, '<h2>', '</h2>')
      expected_header2s = ['Model Details', 'Considerations']
      if expected_graphic.train:
        expected_header2s += ['Train Set']
      if expected_graphic.eval:
        expected_header2s += ['Eval Set']
      if expected_graphic.quantitative:
        expected_header2s += ['Quantitative Analysis']
      self.assertListEqual(header2s, expected_header2s)

    with self.subTest(name='h3'):
      header3s = self.find_all(page, '<h3>', '</h3>')
      expected_header3s = []
      for model_detail in _MODEL_DETAILS_PROPERTIES:
        if mc.model_details.get(model_detail):
          expected_header3s += [model_detail.capitalize()]
      self.assertListEqual(header3s, expected_header3s)

    with self.subTest(name='owners'):
      owners_html = self.find(page, '<h3>Owners</h3>', '<h3>Version</h3>')
      owners = self.parse_ul(owners_html)
      actual_owners = [
          f'{owner["name"]}, {owner["contact"]}'
          for owner in mc.model_details['owners']
      ]
      self.assertSameElements(owners, actual_owners)

    with self.subTest(name='version'):
      version_html = self.find(page, '<h3>Version</h3>', '<h3>License</h3>')
      version_divs = self.find_all(version_html, '<div>', '</div>')
      version_key_vals = [v.split(':') for v in version_divs]
      version_dict = {v[0].strip(): v[1].strip() for v in version_key_vals}
      self.assertDictEqual(
          version_dict, {
              'name': mc.model_details['version']['name'],
              'date': mc.model_details['version']['date'],
              'diff': mc.model_details['version']['diff']
          })

    with self.subTest(name='license'):
      license_value = self.find(page, '<h3>License</h3>', '<h3>References</h3>')
      self.assertEqual(license_value, mc.model_details['license'])

    with self.subTest(name='references'):
      references_html = self.find(page, '<h3>References</h3>',
                                  '<h3>Citation</h3>')
      references = self.parse_ul(references_html)
      self.assertListEqual(references, mc.model_details['references'])

    with self.subTest(name='citation'):
      citation = page.split('<h3>Citation</h3>')[1].strip().split()[0]
      self.assertEqual(citation, mc.model_details['citation'])

    with self.subTest(name='graphics'):
      train_set = self.find(page, '<h2>Train Set</h2>',
                            '<h2>Eval Set</h2>').strip()
      self.assertStartsWith(
          train_set, mc.model_parameters.data.train.graphics.description or '')

      eval_set = self.find(page, '<h2>Eval Set</h2>',
                           '<h2>Quantitative Analysis</h2>').strip()
      self.assertStartsWith(
          eval_set, mc.model_parameters.data.eval.graphics.description or '')

      quantitative_analysis = self.find(page, '<h2>Quantitative Analysis</h2>',
                                        '</div>').strip()
      self.assertStartsWith(quantitative_analysis,
                            mc.quantitative_analysis.graphics.description or '')

  def test_train_data_only(self):
    mc = model_card.ModelCard()
    mc.model_details = {
        'name': 'train model',
        'owners': [{
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }],
        'version': {
            'name': '0.1',
            'date': '2020-01-01',
            'diff': 'Updated dataset.',
        },
        'license': 'Apache 2.0',
        'references': ['https://my_model.xyz.com', 'https://example.com'],
        'citation': 'https://doi.org/foo/bar',
    }
    mc.model_parameters = model_card.ModelParameters(
        model_architecture='knn',
        data=model_card.Data(train=_TRAIN_DATA_WITH_GRAPHIC))
    self.assertRenderedTemplate(mc, ExpectedGraphic(train=True))

  def test_eval_data_only(self):
    mc = model_card.ModelCard()
    mc.model_details = {
        'name': 'eval model',
        'owners': [{
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }],
        'version': {
            'name': '0.2',
            'date': '2020-01-01',
            'diff': 'Updated dataset.',
        },
        'license': 'Apache 2.0',
        'references': ['https://my_model.xyz.com', 'https://example.com'],
        'citation': 'https://doi.org/foo/bar',
    }
    mc.model_parameters = model_card.ModelParameters(
        model_architecture='knn',
        data=model_card.Data(eval=_EVAL_DATA_WITH_GRAPHIC))
    self.assertRenderedTemplate(mc, ExpectedGraphic(eval=True))

  def test_train_and_eval_data(self):
    mc = model_card.ModelCard()
    mc.model_details = {
        'name': 'train and eval model',
        'owners': [{
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }],
        'version': {
            'name': '0.3',
            'date': '2020-01-01',
            'diff': 'Updated dataset.',
        },
        'license': 'Apache 2.0',
        'references': ['https://my_model.xyz.com', 'https://example.com'],
        'citation': 'https://doi.org/foo/bar',
    }
    mc.model_parameters = model_card.ModelParameters(
        model_architecture='knn',
        data=model_card.Data(train=_TRAIN_DATA, eval=_EVAL_DATA))
    self.assertRenderedTemplate(mc, ExpectedGraphic())

  def test_quantitiative_analysis(self):
    mc = model_card.ModelCard()
    mc.model_details = {
        'name': 'quantitative analysis',
        'overview': 'This demonstrates a quantitative analysis graphic.',
        'owners': [{
            'name': 'bar',
            'contact': 'bar@xyz.com'
        }],
        'version': {
            'name': '0.4',
            'date': '2020-01-01',
            'diff': 'Updated dataset.',
        },
        'license': 'Apache 2.0',
        'references': ['https://my_model.xyz.com', 'https://example.com'],
        'citation': 'https://doi.org/foo/bar',
    }
    mc.model_parameters = model_card.ModelParameters(model_architecture='knn')
    mc.quantitative_analysis = model_card.QuantitativeAnalysis(
        graphics=model_card.Graphics(
            description='quantitative analysis graphic',
            collection=[{
                'name': 'quantitative analysis graphic',
                'image': 'I am an image beep boop'
            }]))
    self.assertRenderedTemplate(mc, ExpectedGraphic(quantitative=True))


if __name__ == '__main__':
  absltest.main()
