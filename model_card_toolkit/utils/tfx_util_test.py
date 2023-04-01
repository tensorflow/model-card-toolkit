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

import os
import uuid

import ml_metadata as mlmd
import tensorflow_model_analysis as tfma
from absl.testing import absltest
from ml_metadata.proto import metadata_store_pb2

from model_card_toolkit.model_card import ModelCard, PerformanceMetric
from model_card_toolkit.utils import tfx_util
from model_card_toolkit.utils.testdata import testdata_utils

_SLICING_METRICS = [
    (
        (('weekday', 0), ), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 0.07875693589448929
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.5100112557411194,
                            'lower_bound': 0.4100112557411194,
                            'upper_bound': 0.6100112557411194
                        }
                    }
                }
            }
        }
    ),
    (
        (('weekday', 1), ), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 4.4887189865112305
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4839990735054016,
                            'lower_bound': 0.3839990735054016,
                            'upper_bound': 0.5839990735054016
                        }
                    }
                }
            }
        }
    ),
    (
        (('weekday', 2), ), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 2.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.3767518997192383,
                            'lower_bound': 0.1767518997192383,
                            'upper_bound': 0.5767518997192383
                        }
                    }
                }
            }
        }
    ),
    (
        (('gender', 'male'), ('age', 10)), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 2.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.3767518997192383,
                            'lower_bound': 0.1767518997192383,
                            'upper_bound': 0.5767518997192383
                        }
                    }
                }
            }
        }
    ),
    (
        (('gender', 'female'), ('age', 20)), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 2.092138290405273
                    },
                    'prediction/mean': {
                        'doubleValue': 0.3767518997192383
                    }
                }
            }
        }
    ),
    (
        (), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 1.092138290405273
                    },
                    'prediction/mean': {
                        'boundedValue': {
                            'value': 0.4767518997192383,
                            'lower_bound': 0.2767518997192383,
                            'upper_bound': 0.6767518997192383
                        }
                    },
                    'int_array': {
                        'arrayValue': {
                            'dataType': 'INT32',
                            'int32Values': [1, 2, 3]
                        }
                    },
                    'float_array': {
                        'arrayValue': {
                            'dataType': 'FLOAT32',
                            'float32Values': [1.1, 2.2, 3.3]
                        }
                    },
                    'invalid_array': {
                        'arrayValue': {
                            'dataType': 'UNKNOWN'
                        }
                    }
                }
            }
        }
    )
]

_DATASET_FEATURES = [
    'data_channel', 'date', 'slug', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03',
    'LDA_04', 'abs_title_sentiment_polarity', 'abs_title_subjectivity',
    'average_token_length', 'avg_negative_polarity', 'avg_positive_polarity',
    'global_rate_negative_words', 'global_rate_positive_words',
    'global_sentiment_polarity', 'global_subjectivity', 'is_weekend',
    'kw_avg_avg', 'kw_avg_max', 'kw_avg_min', 'kw_max_avg', 'kw_max_max',
    'kw_max_min', 'kw_min_avg', 'kw_min_max', 'kw_min_min',
    'max_negative_polarity', 'max_positive_polarity', 'min_negative_polarity',
    'min_positive_polarity', 'n_hrefs', 'n_imgs', 'n_keywords',
    'n_non_stop_unique_tokens', 'n_non_stop_words', 'n_self_hrefs',
    'n_shares_percentile', 'n_tokens_content', 'n_tokens_title',
    'n_unique_tokens', 'n_videos', 'rate_negative_words',
    'rate_positive_words', 'self_reference_avg_shares',
    'self_reference_max_shares', 'self_reference_min_shares', 'timedelta',
    'title_sentiment_polarity', 'title_subjectivity', 'weekday'
]


class TfxUtilsTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.tmp_db_path = os.path.join(
        absltest.get_default_test_tmpdir(), f'test_mlmd_{uuid.uuid4()}.db'
    )

  def _get_empty_metadata_store(self):
    """Returns an empty in memory mlmd store."""
    empty_db_config = metadata_store_pb2.ConnectionConfig()
    empty_db_config.fake_database.SetInParent()
    return mlmd.MetadataStore(empty_db_config)

  def test_get_metrics_artifacts_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    got_metrics = tfx_util.get_metrics_artifacts_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
    )
    got_metrics_ids = [a.id for a in got_metrics]
    self.assertCountEqual(
        got_metrics_ids, testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS
    )

  def test_get_metrics_artifacts_for_model_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.get_metrics_artifacts_for_model(store, model.id)

  def test_get_metrics_artifacts_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.get_metrics_artifacts_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID
      )

  def test_get_metrics_artifacts_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.get_metrics_artifacts_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
      )

  def test_get_stats_artifacts_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    got_stats = tfx_util.get_stats_artifacts_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
    )
    got_stats_ids = [a.id for a in got_stats]
    self.assertCountEqual(
        got_stats_ids, [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID]
    )

  def test_get_stats_artifacts_for_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.get_stats_artifacts_for_model(store, model.id)

  def test_get_stats_artifacts_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.get_stats_artifacts_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID
      )

  def test_get_stats_artifacts_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.get_stats_artifacts_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
      )

  def test_generate_model_card_for_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    model_card = tfx_util.generate_model_card_for_model(
        store, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
    )
    trainers = store.get_executions_by_id([testdata_utils.TFX_0_21_TRAINER_ID])
    self.assertNotEmpty(trainers)
    model_details = model_card.model_details
    self.assertEqual(
        model_details.name, trainers[-1].properties['module_file'].string_value
    )
    self.assertEqual(
        model_details.version.name,
        trainers[-1].properties['checksum_md5'].string_value
    )
    self.assertIn(
        trainers[-1].properties['pipeline_name'].string_value,
        [reference.reference for reference in model_details.references]
    )

    datasets = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_MODEL_DATASET_ID]
    )
    self.assertNotEmpty(datasets)

  def test_generate_model_card_for_model_with_model_not_found(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'model_id cannot be found'):
      model = metadata_store_pb2.Artifact()
      tfx_util.generate_model_card_for_model(store, model.id)

  def test_generate_model_card_for_model_with_invalid_model(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    with self.assertRaisesRegex(ValueError, 'not an instance of Model'):
      tfx_util.generate_model_card_for_model(
          store, testdata_utils.TFX_0_21_MODEL_DATASET_ID
      )

  def test_generate_model_card_for_model_with_invalid_db(self):
    empty_db = self._get_empty_metadata_store()
    with self.assertRaisesRegex(ValueError, '`store` is invalid'):
      tfx_util.generate_model_card_for_model(
          empty_db, testdata_utils.TFX_0_21_MODEL_ARTIFACT_ID
      )

  def test_read_stats_protos(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID]
    )
    self.assertLen(stats, 1)
    data_stats = tfx_util.read_stats_protos(stats[-1].uri)
    self.assertLen(data_stats, 2)  # Split-eval, Split-train

  def test_read_stats_proto(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID]
    )
    self.assertLen(stats, 1)
    train_stats = tfx_util.read_stats_proto(stats[-1].uri, 'Split-train')
    self.assertIsNotNone(train_stats)
    eval_stats = tfx_util.read_stats_proto(stats[-1].uri, 'Split-eval')
    self.assertIsNotNone(eval_stats)

  def test_read_stats_proto_with_invalid_split(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID]
    )
    self.assertLen(stats, 1)
    actual_stats = tfx_util.read_stats_proto(stats[-1].uri, 'invalid_split')
    self.assertIsNone(actual_stats)

  def test_read_stats_proto_with_invalid_uri(self):
    self.assertIsNone(tfx_util.read_stats_proto('/does/not/exist/', 'train'))
    self.assertIsNone(tfx_util.read_stats_proto('/does/not/exist/', 'eval'))

  def test_filter_features(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    stats = store.get_artifacts_by_id(
        [testdata_utils.TFX_0_21_STATS_ARTIFACT_ID]
    )
    dataset_stats = tfx_util.read_stats_protos(stats[-1].uri)[0].datasets[0]

    one_half_of_the_features = _DATASET_FEATURES[:27]
    the_other_half_of_the_features = _DATASET_FEATURES[27:]

    with self.subTest(name='features_include'):
      filtered_features = [
          feature.path.step[0] for feature in tfx_util.filter_features(
              dataset_stats, features_include=one_half_of_the_features
          ).features
      ]
      self.assertSameElements(one_half_of_the_features, filtered_features)
    with self.subTest(name='features_exclude'):
      filtered_features = [
          feature.path.step[0] for feature in tfx_util.filter_features(
              dataset_stats, features_exclude=one_half_of_the_features
          ).features
      ]
      self.assertSameElements(
          the_other_half_of_the_features, filtered_features
      )
    with self.subTest(
        name='both features_include and features_exclude (invalid)'
    ):
      with self.assertRaises(ValueError):
        tfx_util.filter_features(
            dataset_stats, features_include=one_half_of_the_features,
            features_exclude=the_other_half_of_the_features
        )
    with self.subTest(
        name='neither features_include nor features_exclude (invalid)'
    ):
      with self.assertRaises(ValueError):
        tfx_util.filter_features(dataset_stats)

  def test_read_metrics_eval_result(self):
    store = testdata_utils.get_tfx_pipeline_metadata_store(self.tmp_db_path)
    metrics = store.get_artifacts_by_id(
        testdata_utils.TFX_0_21_METRICS_ARTIFACT_IDS
    )
    eval_result = tfx_util.read_metrics_eval_result(metrics[-1].uri)
    self.assertIsNotNone(eval_result)

  def test_read_metrics_eval_result_with_invalid_uri(self):
    self.assertIsNone(tfx_util.read_metrics_eval_result('/does/not/exist/'))

  def test_annotate_eval_results_metrics(self):
    eval_result = tfma.EvalResult(  # pytype: disable=wrong-arg-types
        slicing_metrics=_SLICING_METRICS,
        plots=None,
        attributions=None,
        config=None,
        data_location=None,
        file_format=None,
        model_location=None)
    model_card = ModelCard()
    tfx_util.annotate_eval_result_metrics(model_card, eval_result)

    expected_metrics = [
        PerformanceMetric(
            type='average_loss', value='0.07875693589448929', slice='weekday_0'
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.5100112557411194',
            slice='weekday_0'
        ),
        PerformanceMetric(
            type='average_loss', value='4.4887189865112305', slice='weekday_1'
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.4839990735054016',
            slice='weekday_1'
        ),
        PerformanceMetric(
            type='average_loss', value='2.092138290405273', slice='weekday_2'
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.3767518997192383',
            slice='weekday_2'
        ),
        PerformanceMetric(
            type='average_loss', value='2.092138290405273',
            slice='gender_male_X_age_10'
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.3767518997192383',
            slice='gender_male_X_age_10'
        ),
        PerformanceMetric(
            type='average_loss', value='2.092138290405273',
            slice='gender_female_X_age_20'
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.3767518997192383',
            slice='gender_female_X_age_20'
        ),
        PerformanceMetric(
            type='average_loss', value='1.092138290405273', slice=''
        ),
        PerformanceMetric(
            type='prediction/mean', value='0.4767518997192383', slice=''
        ),
        PerformanceMetric(type='int_array', value='1, 2, 3', slice=''),
        PerformanceMetric(type='float_array', value='1.1, 2.2, 3.3', slice='')
    ]
    self.assertEqual(
        len(model_card.quantitative_analysis.performance_metrics),
        len(expected_metrics)
    )
    for actual_metric, expected_metric in zip(
        model_card.quantitative_analysis.performance_metrics, expected_metrics
    ):
      self.assertEqual(actual_metric.type, expected_metric.type)
      self.assertEqual(actual_metric.slice, expected_metric.slice)
      self.assertEqual(actual_metric.value, expected_metric.value)

  def test_filter_metrics(self):
    eval_result = tfma.EvalResult(  # pytype: disable=wrong-arg-types
        slicing_metrics=_SLICING_METRICS,
        plots=None,
        attributions=None,
        config=None,
        data_location=None,
        file_format=None,
        model_location=None)
    metrics_include = ['average_loss']
    metrics_exclude = [
        'prediction/mean', 'int_array', 'float_array', 'invalid_array'
    ]
    expected_slicing_metrics = [
        (
            (('weekday', 0), ), {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 0.07875693589448929
                        }
                    }
                }
            }
        ),
        (
            (('weekday', 1), ), {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 4.4887189865112305
                        }
                    }
                }
            }
        ),
        (
            (('weekday', 2), ), {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 2.092138290405273
                        }
                    }
                }
            }
        ),
        (
            (('gender', 'male'), ('age', 10)), {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 2.092138290405273
                        }
                    }
                }
            }
        ),
        (
            (('gender', 'female'), ('age', 20)), {
                '': {
                    '': {
                        'average_loss': {
                            'doubleValue': 2.092138290405273
                        }
                    }
                }
            }
        ),
        ((), {
            '': {
                '': {
                    'average_loss': {
                        'doubleValue': 1.092138290405273
                    }
                }
            }
        })
    ]
    with self.subTest(name='metrics_include'):
      self.assertEqual(
          tfx_util.filter_metrics(
              eval_result, metrics_include=metrics_include
          ).slicing_metrics, expected_slicing_metrics
      )
    with self.subTest(name='metrics_exclude'):
      self.assertEqual(
          tfx_util.filter_metrics(
              eval_result, metrics_exclude=metrics_exclude
          ).slicing_metrics, expected_slicing_metrics
      )
    with self.subTest(
        name='both metrics_include and metrics_exclude (invalid)'
    ):
      with self.assertRaises(ValueError):
        tfx_util.filter_metrics(
            eval_result, metrics_include=metrics_include,
            metrics_exclude=metrics_exclude
        )
    with self.subTest(
        name='neither metrics_include nor metrics_exclude (invalid)'
    ):
      with self.assertRaises(ValueError):
        tfx_util.filter_metrics(eval_result)


if __name__ == '__main__':
  absltest.main()
