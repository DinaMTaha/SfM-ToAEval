import os
import re
import sys
import numpy as np
import glob
import time
import logging
import subprocess
from .database import Database
from ..experiment import Experiment


class Dataset:
    def __init__(self, path, K, image_format='jpg'):
        self.path = path
        self.K = K
        self.image_format = image_format
        self.sequences = {}

    def load(self):
        for sequence in glob.glob(os.path.join(self.path, '*/')):
            self.sequences[sequence.split(os.sep)[-2]] = sequence

    def evaluate(self, database, databases_path, point_clouds_path, detector_types, feature_types):
        for sequence, folder in self.sequences.items():
            print(f' {sequence} '.center(60, '='))
            for detector_type in detector_types:
                for feature_type in feature_types:
                    print(f' {detector_type}_{feature_type} '.center(60, '-'))
                    try:
                        point_cloud_path = os.path.join(
                            point_clouds_path, f'{sequence}_{detector_type}_{feature_type}')
                        if os.path.exists(os.path.join(point_cloud_path, '0')):
                            continue
                        if not os.path.exists(point_cloud_path):
                            os.makedirs(point_cloud_path)
                        image_path = os.path.join(folder, 'images')
                        experiment = Experiment(self.K,
                                                database,
                                                sequence,
                                                image_path,
                                                self.image_format,
                                                detector_type,
                                                feature_type,
                                                )
                        tic = time.time()
                        experiment.createViews()
                        toc = time.time()
                        logging.info('Processed %d views in %f seconds',
                                     len(experiment.views), toc - tic)
                        tic = time.time()
                        experiment.createMatches()
                        toc = time.time()
                        logging.info('Processed %d pairs in %f seconds',
                                     len(experiment.matches), toc - tic)
                        colmap_db_path = os.path.join(
                            databases_path, f'{sequence}_{detector_type}_{feature_type}.sqlite')
                        if os.path.exists(colmap_db_path):
                            os.remove(colmap_db_path)
                        experiment.toColmap(colmap_db_path)
                        colmap_log = bytes()
                        ret = subprocess.check_output([
                            'colmap',
                            'feature_extractor',
                            '--database_path', colmap_db_path,
                            '--image_path', image_path,
                            '--SiftExtraction.use_gpu', '0',
                        ],
                            stderr=subprocess.PIPE)
                        ret = subprocess.check_output([
                            'colmap',
                            'exhaustive_matcher',
                            '--database_path', colmap_db_path,
                            '--SiftMatching.use_gpu', '0',
                        ],
                            stderr=subprocess.PIPE)
                        ret = subprocess.check_output([
                            'colmap',
                            'mapper',
                            '--database_path', colmap_db_path,
                            '--image_path', image_path,
                            '--output_path', point_cloud_path,
                        ],
                            stderr=subprocess.PIPE)
                        colmap_log += ret
                        ret = subprocess.check_output([
                            'colmap',
                            'model_analyzer',
                            '--path', os.path.join(point_cloud_path, '0'),
                        ],
                            stderr=subprocess.PIPE)
                        analysis = {}
                        for line in ret.decode('utf-8').strip().split('\n'):
                            key, value = line.split(':')
                            analysis[key.strip().lower()
                                     ] = value.strip().lower()
                        database.putAnalysis(experiment.sequence_id,
                                             experiment.detector_type_id,
                                             experiment.feature_type_id,
                                             int(analysis['points']),
                                             int(analysis['observations']),
                                             float(analysis['mean reprojection error'].replace('px', '')))
                        colmap_log += ret
                        ret = subprocess.check_output([
                            'colmap',
                            'model_converter',
                            '--input_path', os.path.join(
                                point_cloud_path, '0'),
                            '--output_path', os.path.join(
                                point_cloud_path, '0.ply'),
                            '--output_type', 'PLY',
                        ],
                            stderr=subprocess.PIPE)
                        print(re.findall(
                            r'Initializing with image pair #\d+ and #\d+', colmap_log.decode('utf-8'))[-1])
                        print(re.findall(
                            r'Elapsed time: \d+.\d+ \[minutes\]', colmap_log.decode('utf-8'))[-1])
                        print(re.findall(r'Points: \d+',
                              colmap_log.decode('utf-8'))[-1])
                        print(re.findall(
                            r'Mean reprojection error: \d+.\d+px', colmap_log.decode('utf-8'))[-1])
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(e)
                    except:
                        print(sys.exc_info())
                    database.commit()
