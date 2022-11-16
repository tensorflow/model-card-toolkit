"""Tests for model_card_toolkit.tfx"""
from absl.testing import absltest


class TfxModuleTest(absltest.TestCase):

  def test_error_on_import(self):
    with self.assertRaises(ImportError):
      from model_card_toolkit.tfx.component import ModelCardGenerator

if __name__ == '__main__':
  absltest.main()
