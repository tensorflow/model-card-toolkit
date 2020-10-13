"""Tests for model_card_toolkit.documentation.examples.cats_vs_dogs."""

from absl.testing import absltest
from model_card_toolkit.documentation.examples import cats_vs_dogs

SLICES = ['combined', 'cat', 'dog']
FIELDS = ['examples', 'labels']


class CatsVsDogsTest(absltest.TestCase):

  def test_get_data(self):
    data = cats_vs_dogs.get_data()
    self.assertSameElements(data.keys(), SLICES)
    self.assertSameElements(data['combined'].keys(), FIELDS)
    self.assertSameElements(data['cat'].keys(), FIELDS)
    self.assertSameElements(data['dog'].keys(), FIELDS)
    self.assertLen(data['combined']['labels'],
                   len(data['combined']['examples']))
    self.assertLen(data['dog']['labels'], len(data['dog']['examples']))
    self.assertLen(data['cat']['labels'], len(data['cat']['examples']))
    self.assertLen(data['combined']['labels'], 320)
    self.assertLen(data['cat']['labels'], 149)
    self.assertLen(data['dog']['labels'], 171)
    self.assertLen(data['cat']['labels'], sum(data['combined']['labels']))
    self.assertLen(
        data['dog']['labels'],
        len(data['combined']['labels']) - sum(data['combined']['labels']))


if __name__ == '__main__':
  absltest.main()
