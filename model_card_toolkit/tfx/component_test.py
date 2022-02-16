"""Tests for model_card_toolkit.tfx.component."""

import json as json_lib

from absl.testing import absltest

from model_card_toolkit.tfx import artifact
from model_card_toolkit.tfx.component import ModelCardGenerator

from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(absltest.TestCase):

  def test_component_construction(self):
    this_component = ModelCardGenerator(
        statistics=channel_utils.as_channel(
            [standard_artifacts.ExampleStatistics()]),
        evaluation=channel_utils.as_channel(
            [standard_artifacts.ModelEvaluation()]),
        pushed_model=channel_utils.as_channel(
            [standard_artifacts.PushedModel()]),
        json=json_lib.dumps(
            {'model_details': {
                'name': 'my model',
                'version': {
                    'name': 'v1'
                }
            }}),
        template_io=[('path/to/html/template', 'mc.html'),
                     ('path/to/md/template', 'mc.md')])

    with self.subTest('outputs'):
      self.assertEqual(this_component.outputs['model_card'].type_name,
                       artifact.ModelCard.TYPE_NAME)

    with self.subTest('exec_properties'):
      self.assertDictEqual(
          {
              'json': json_lib.dumps({
                  'model_details': {
                      'name': 'my model',
                      'version': {
                          'name': 'v1'
                      }
                  }
              }),
              'template_io': [('path/to/html/template', 'mc.html'),
                              ('path/to/md/template', 'mc.md')]
          }, this_component.exec_properties)

  def test_empty_component_construction(self):
    this_component = ModelCardGenerator()
    with self.subTest('outputs'):
      self.assertEqual(this_component.outputs['model_card'].type_name,
                       artifact.ModelCard.TYPE_NAME)


if __name__ == '__main__':
  absltest.main()
