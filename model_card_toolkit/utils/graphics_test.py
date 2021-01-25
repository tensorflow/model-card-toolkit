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
"""Tests for the TFX-OSS pipeline utilities."""

import logging
from absl.testing import absltest
from absl.testing import parameterized
from model_card_toolkit import model_card as model_card_module
from model_card_toolkit.utils import graphics
import tensorflow_model_analysis as tfma
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2


class GraphicsTest(parameterized.TestCase):

  def assertGraphEqual(self, g: graphics._Graph, h: graphics._Graph):
    self.assertSequenceEqual(g.x, h.x)
    self.assertSequenceEqual(g.y, h.y)
    if g.xerr and h.xerr:
      self.assertSequenceEqual(g.xerr, h.xerr)
    else:
      self.assertEqual(g.xerr, h.xerr)
    self.assertEqual(g.xlabel, h.xlabel)
    self.assertEqual(g.ylabel, h.ylabel)
    self.assertEqual(g.title, h.title)
    self.assertEqual(g.name, h.name)
    self.assertEqual(g.color, h.color)

  def test_generate_graph_from_feature_statistics(self):
    numeric_feature_stats = text_format.Parse(
        """
        path {
          step: "numeric_feature"
        }
        type: INT
        num_stats {
          histograms {
            buckets {
              low_value: 0.0
              high_value: 50.0
              sample_count: 6.0
            }
            buckets {
              low_value: 50.0
              high_value: 100.0
              sample_count: 4.0
            }
          }
        }""", statistics_pb2.FeatureNameStatistics())
    self.assertGraphEqual(
        graphics._generate_graph_from_feature_statistics(numeric_feature_stats),
        graphics._Graph(
            x=[6, 4],
            y=['0.00-50.00', '50.00-100.00'],
            xlabel='counts',
            ylabel='buckets',
            title='counts | numeric_feature',
            name='counts | numeric_feature'))

    string_feature_stats = text_format.Parse(
        """
        path {
          step: "string_feature"
        }
        type: STRING
        string_stats {
          rank_histogram {
            buckets {
              label: 'News'
              sample_count: 1387.0
            }
            buckets {
              label: 'Tech'
              sample_count: 3395.0
            }
            buckets {
              label: 'Sports'
              sample_count: 2395.0
            }
          }
        }""", statistics_pb2.FeatureNameStatistics())
    self.assertGraphEqual(
        graphics._generate_graph_from_feature_statistics(string_feature_stats),
        graphics._Graph(
            x=[1387, 3395, 2395],
            y=['News', 'Tech', 'Sports'],
            xlabel='counts',
            ylabel='buckets',
            title='counts | string_feature',
            name='counts | string_feature'))

    bytes_feature_stats = text_format.Parse(
        """
        path {
          step: "bytes_feature"
        }
        type: BYTES
        bytes_stats {}""", statistics_pb2.FeatureNameStatistics())
    self.assertIsNone(
        graphics._generate_graph_from_feature_statistics(bytes_feature_stats))

    struct_feature_stats = text_format.Parse(
        """
        path {
          step: "struct_feature"
        }
        type: STRUCT
        struct_stats {}""", statistics_pb2.FeatureNameStatistics())
    self.assertIsNone(
        graphics._generate_graph_from_feature_statistics(struct_feature_stats))

  def test_annotate_dataset_feature_statistics_plots(self):
    train_stats = text_format.Parse(
        """
    datasets {
      features {
        path {
          step: "LDA_00"
        }
        type: FLOAT
        num_stats {
          histograms {
            buckets {
              low_value: 0.0
              high_value: 100.0
              sample_count: 10.0
            }
          }
          histograms {
            buckets {
              low_value: 0.0
              high_value: 50.0
              sample_count: 4.0
            }
            buckets {
              low_value: 50.0
              high_value: 100.0
              sample_count: 4.0
            }
            type: QUANTILES
          }
        }
      }
      features {
        path {
          step: "LDA_01"
        }
        type: FLOAT
        num_stats {
          histograms {
            buckets {
              low_value: 0.0
              high_value: 100.0
              sample_count: 10.0
            }
          }
          histograms {
            buckets {
              low_value: 0.0
              high_value: 50.0
              sample_count: 4.0
            }
            buckets {
              low_value: 50.0
              high_value: 100.0
              sample_count: 4.0
            }
            type: QUANTILES
          }
        }
      }
      features {
        path {
          step: "LDA_02"
        }
        type: FLOAT
        num_stats {
          histograms {
            buckets {
              low_value: 0.0
              high_value: 100.0
              sample_count: 10.0
            }
          }
          histograms {
            buckets {
              low_value: 0.0
              high_value: 50.0
              sample_count: 4.0
            }
            buckets {
              low_value: 50.0
              high_value: 100.0
              sample_count: 4.0
            }
            type: QUANTILES
          }
        }
      }
      features {
        path {
          step: "LDA_03"
        }
        type: STRING
        bytes_stats {
          unique: 1
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())
    eval_stats = text_format.Parse(
        """
    datasets {
      features {
        path {
          step: "data_channel"
        }
        type: STRING
        string_stats {
          rank_histogram {
            buckets {
              label: 'News'
              sample_count: 1387.0
            }
            buckets {
              label: 'Tech'
              sample_count: 3395.0
            }
            buckets {
              label: 'Sports'
              sample_count: 2395.0
            }
          }
        }
      }
      features {
        path {
          step: "date"
        }
        type: STRING
        string_stats {
          rank_histogram {
            buckets {
              label: '2014-12-10'
              sample_count: 40.0
            }
            buckets {
              label: '2014-11-06'
              sample_count: 37.0
            }
          }
        }
      }
      features {
        path {
          step: "slug"
        }
        type: STRING
        string_stats {
          rank_histogram {
            buckets {
              label: 'zynga-q3-earnings'
              sample_count: 1.0
            }
            buckets {
              label: 'zumba-ad'
              sample_count: 1.0
            }
          }
        }
      }
    }
    """, statistics_pb2.DatasetFeatureStatisticsList())

    model_card = model_card_module.ModelCard()
    graphics.annotate_dataset_feature_statistics_plots(model_card, train_stats,
                                                       eval_stats)

    expected_plot_names_train = {
        'counts | LDA_00', 'counts | LDA_01', 'counts | LDA_02'
    }
    expected_plot_names_eval = {
        'counts | data_channel', 'counts | date', 'counts | slug'
    }

    self.assertSameElements([
        g.name
        for g in model_card.model_parameters.data.train.graphics.collection
    ], expected_plot_names_train)
    self.assertSameElements([
        g.name
        for g in model_card.model_parameters.data.eval.graphics.collection
    ], expected_plot_names_eval)

    graphs = model_card.model_parameters.data.train.graphics.collection + model_card.model_parameters.data.eval.graphics.collection
    for graph in graphs:
      logging.info('%s: %s', graph.name, graph.image)
      self.assertNotEmpty(graph.image, f'feature {graph.name} has empty plot')

  def test_generate_graph_from_slicing_metrics(self):
    slicing_metrics = [
        ((('weekday', 0),), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 0.07875693589448929
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.5100112557411194,
                            'lowerBound': 0.4100112557411194,
                            'upperBound': 0.6100112557411194,
                        }
                    }
                }
            }
        }),
        ((('weekday', 1),), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 4.4887189865112305
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4839990735054016,
                            'lowerBound': 0.3839990735054016,
                            'upperBound': 0.5839990735054016,
                        }
                    }
                }
            }
        }),
        (
            (('weekday', 2),),
            {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 2.092138290405273
                        },
                        'prediction/mean': {
                            'doubleValue': 0.3767518997192383
                        },
                        '__ERROR__': {
                            # CI not computed because only 16 samples
                            # were non-empty. Expected 20.
                            'bytesValue':
                                'Q0kgbm90IGNvbXB1dGVkIGJlY2F1c2Ugb25seSAxNiBzYW1wbGVzIHdlcmUgbm9uLWVtcHR5LiBFeHBlY3RlZCAyMC4='
                        }
                    }
                }
            }),
        ((), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 1.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4767518997192383,
                            'lowerBound': 0.2767518997192383,
                            'upperBound': 0.6767518997192383,
                        }
                    }
                }
            }
        })
    ]
    self.assertGraphEqual(
        graphics._generate_graph_from_slicing_metrics(slicing_metrics,
                                                      'average_loss'),
        graphics._Graph(
            x=[
                0.07875693589448929, 4.4887189865112305, 2.092138290405273,
                1.092138290405273
            ],
            y=['0', '1', '2', 'Overall'],
            xlabel='average_loss',
            ylabel='slices',
            title='average_loss',
            name='average_loss',
            color='#A142F4'))
    self.assertGraphEqual(
        graphics._generate_graph_from_slicing_metrics(slicing_metrics,
                                                      'average_loss',
                                                      'weekday'),
        graphics._Graph(
            x=[
                0.07875693589448929, 4.4887189865112305, 2.092138290405273,
                1.092138290405273
            ],
            y=['0', '1', '2', 'Overall'],
            xlabel='average_loss',
            ylabel='slices',
            title='average_loss | weekday',
            name='average_loss | weekday',
            color='#A142F4'))
    self.assertGraphEqual(
        graphics._generate_graph_from_slicing_metrics(slicing_metrics,
                                                      'prediction/mean'),
        graphics._Graph(
            x=[
                0.5100112557411194, 0.4839990735054016, 0.3767518997192383,
                0.4767518997192383
            ],
            y=['0', '1', '2', 'Overall'],
            xerr=[[0.09999999999999998, 0.10000000000000003, 0, 0.2],
                  [0.09999999999999998, 0.09999999999999998, 0, 0.2]],
            xlabel='prediction/mean',
            ylabel='slices',
            title='prediction/mean',
            name='prediction/mean',
            color='#A142F4'))

    self.assertGraphEqual(
        graphics._generate_graph_from_slicing_metrics(slicing_metrics,
                                                      'prediction/mean',
                                                      'weekday'),
        graphics._Graph(
            x=[
                0.5100112557411194, 0.4839990735054016, 0.3767518997192383,
                0.4767518997192383
            ],
            y=['0', '1', '2', 'Overall'],
            xerr=[[0.09999999999999998, 0.10000000000000003, 0, 0.2],
                  [0.09999999999999998, 0.09999999999999998, 0, 0.2]],
            xlabel='prediction/mean',
            ylabel='slices',
            title='prediction/mean | weekday',
            name='prediction/mean | weekday',
            color='#A142F4'))

  def test_annotate_eval_results_plots(self):
    slicing_metrics = [
        ((('weekday', 0),), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 0.07875693589448929
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.5100112557411194,
                            'lowerBound': 0.4100112557411194,
                            'upperBound': 0.6100112557411194,
                        }
                    },
                    'average_loss_diff': {}
                }
            }
        }),
        ((('weekday', 1),), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 4.4887189865112305
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4839990735054016,
                            'lowerBound': 0.3839990735054016,
                            'upperBound': 0.5839990735054016,
                        }
                    },
                    'average_loss_diff': {}
                }
            }
        }),
        ((('weekday', 2),), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 2.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.3767518997192383,
                            'lowerBound': 0.1767518997192383,
                            'upperBound': 0.5767518997192383,
                        }
                    },
                    'average_loss_diff': {}
                }
            }
        }),
        ((('gender', 'male'), ('age', 10)), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 2.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.3767518997192383,
                            'lowerBound': 0.1767518997192383,
                            'upperBound': 0.5767518997192383,
                        }
                    },
                    'average_loss_diff': {}
                }
            }
        }),
        (
            (('gender', 'female'), ('age', 20)),
            {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 2.092138290405273
                        },
                        'prediction/mean': {
                            'doubleValue': 0.3767518997192383
                        },
                        'average_loss_diff': {},
                        '__ERROR__': {
                            # CI not computed because only 16 samples
                            # were non-empty. Expected 20.
                            'bytesValue':
                                'Q0kgbm90IGNvbXB1dGVkIGJlY2F1c2Ugb25seSAxNiBzYW1wbGVzIHdlcmUgbm9uLWVtcHR5LiBFeHBlY3RlZCAyMC4='
                        }
                    }
                }
            }),
        ((), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 1.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4767518997192383,
                            'lowerBound': 0.2767518997192383,
                            'upperBound': 0.6767518997192383,
                        }
                    },
                    'average_loss_diff': {}
                }
            }
        })
    ]
    eval_result = tfma.EvalResult(
        slicing_metrics=slicing_metrics,
        plots=None,
        attributions=None,
        config=None,
        data_location=None,
        file_format=None,
        model_location=None)
    model_card = model_card_module.ModelCard()
    graphics.annotate_eval_result_plots(model_card, eval_result)

    expected_metrics_names = {
        'average_loss | weekday', 'prediction/mean | weekday',
        'average_loss | gender, age', 'prediction/mean | gender, age'
    }
    self.assertSameElements(
        expected_metrics_names,
        [g.name for g in model_card.quantitative_analysis.graphics.collection])

    for graph in model_card.quantitative_analysis.graphics.collection:
      logging.info('%s: %s', graph.name, graph.image)
      self.assertNotEmpty(graph.image, f'feature {graph.name} has empty plot')

  @parameterized.parameters([
      [(), ('Overall', 'Overall')],
      [(('gender', 'male'),), ('gender', 'male')],
      [(
          ('gender', 'male'),
          ('zip', 12345),
      ), ('gender, zip', 'male, 12345')],
      [(
          ('gender', 'male'),
          ('zip', 12345),
          ('height', 5.7),
      ), ('gender, zip, height', 'male, 12345, 5.7')],
      [(('gender', 'male'), ('zip', 12345), ('height', 5.7), ('comment',
                                                              u'你好')),
       ('gender, zip, height, comment', u'male, 12345, 5.7, 你好')],
  ])
  def test_stringify_slice_key(self, slices, expected_result):
    result = graphics.stringify_slice_key(slices)
    self.assertEqual(result, expected_result)


if __name__ == '__main__':
  absltest.main()
