import os
import numpy as np
import glob
import sqlite3
from ..colmap.read_write_model import read_model


class Analyzer:
    def __init__(self, database_path, point_cloud_path, analysis_database_path):
        self.database_path = database_path
        self.point_cloud_path = point_cloud_path
        self.analysis_database_path = analysis_database_path

        if os.path.exists(self.analysis_database_path):
            os.remove(self.analysis_database_path)
        con = sqlite3.connect(self.analysis_database_path)
        cur = con.cursor()
        cur.executescript('''
            CREATE TABLE IF NOT EXISTS correspondence (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                sequence TEXT,
                detector_type TEXT,
                feature_type TEXT,
                feature_count INTEGER,
                descriptor_size REAL,
                descriptor_compressed_size REAL,
                extraction_time REAL,
                matches_count REAL,
                matching_time REAL);
            CREATE TABLE IF NOT EXISTS reconstruction (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                sequence TEXT,
                detector_type TEXT,
                feature_type TEXT,
                points INTEGER,
                observations INTEGER,
                error REAL);
            CREATE TABLE IF NOT EXISTS ranking (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                sequence TEXT,
                detector_type1 TEXT,
                feature_type1 TEXT,
                detector_type2 TEXT,
                feature_type2 TEXT,
                points1 INTEGER,
                points2 INTEGER,
                error1 REAL,
                error2 REAL);
            CREATE VIEW IF NOT EXISTS ranking_q AS
                SELECT sequence, detector_type1 AS detector_type, feature_type1 AS feature_type,
                    SUM(CASE WHEN error1 < error2 THEN 1 ELSE 0 END) AS score
                FROM ranking
                WHERE detector_type1 != "None" AND detector_type2 != "None"
                GROUP BY sequence, detector_type, feature_type
                ORDER BY sequence,  SUM(CASE WHEN error1 < error2 THEN 1 ELSE 0 END) DESC;
            CREATE VIEW IF NOT EXISTS ranking_by_feature_q AS
                SELECT feature_type, avg(score) AS score
                FROM ranking_q
                GROUP BY feature_type
                ORDER BY avg(score) DESC;
            CREATE VIEW IF NOT EXISTS ranking_by_detector_q AS
                SELECT detector_type, avg(score) AS score
                FROM ranking_q
                GROUP BY detector_type
                ORDER BY avg(score) DESC;
            CREATE VIEW IF NOT EXISTS ranking_by_feature_detector_q AS
                SELECT detector_type, feature_type, avg(score) AS score
                FROM ranking_q
                GROUP BY detector_type, feature_type
                ORDER BY feature_type, avg(score) DESC;
            CREATE VIEW IF NOT EXISTS best_detector_by_feature_q AS
                SELECT feature_type,
                    first_value(detector_type)
                        over (partition BY feature_type ORDER BY score DESC) AS best_detector,
                    first_value(score)
                        over (partition BY feature_type ORDER BY score DESC) AS best_score
                FROM (
                    SELECT detector_type, feature_type, avg(score) AS score
                    FROM ranking_q
                    GROUP BY detector_type, feature_type
                    ORDER BY avg(score) DESC
                    ) AS r
                GROUP BY feature_type
                ORDER BY feature_type;
            CREATE VIEW IF NOT EXISTS best_feature_by_detector_q AS
                SELECT detector_type,
                    first_value(feature_type)
                        over (partition BY detector_type ORDER BY score DESC) AS best_feature,
                    first_value(score)
                        over (partition BY detector_type ORDER BY score DESC) AS best_score
                FROM (
                    SELECT detector_type, feature_type, avg(score) AS score
                    FROM ranking_q
                    GROUP BY detector_type, feature_type
                    ORDER BY avg(score) DESC
                    ) AS r
                GROUP BY detector_type
                ORDER BY detector_type;
            CREATE VIEW IF NOT EXISTS best_detector_feature_by_sequence_q AS
                SELECT sequence,
                    first_value(detector_type)
                        over (partition BY sequence ORDER BY score DESC) AS best_detector,
                    first_value(feature_type)
                        over (partition BY sequence ORDER BY score DESC) AS best_feature,
                    first_value(score)
                        over (partition BY sequence ORDER BY score DESC) AS best_score
                FROM ranking_q
                GROUP BY sequence
                ORDER BY sequence;
            ''')
        con.commit()
        con.close()

    def analyze(self):
        sequences = {}
        for idx, point_cloud_path in enumerate(sorted(glob.glob(os.path.join(self.point_cloud_path, '*')))):
            sequence, detector_type, feature_type = os.path.split(point_cloud_path)[
                1].split('_')
            if sequence not in sequences:
                sequences[sequence] = {}
            point_cloud_path = os.path.join(point_cloud_path, '0')
            if os.path.exists(point_cloud_path):
                _, _, points3D = read_model(point_cloud_path, ext='.bin')
                sequences[sequence][(detector_type, feature_type)] = sorted(
                    [point.error for point in points3D.values()])
            else:
                sequences[sequence][(detector_type, feature_type)] = [np.inf]
        con = sqlite3.connect(self.analysis_database_path)
        cur = con.cursor()
        for sequence in sequences:
            for key1 in sequences[sequence]:
                for key2 in sequences[sequence]:
                    if key1 != key2:
                        points1 = sequences[sequence][key1]
                        points2 = sequences[sequence][key2]
                        min_len = min(len(points1), len(points2))
                        points1 = points1[:min_len]
                        points2 = points2[:min_len]
                        error1 = sum(points1)/min_len
                        error2 = sum(points2)/min_len
                        cur.execute('''
                            INSERT INTO ranking(sequence,
                                detector_type1, feature_type1,
                                detector_type2, feature_type2,
                                points1, points2,
                                error1, error2)
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
                            ''', (sequence, key1[0], key1[1], key2[0], key2[1],
                                  len(points1), len(points2), error1, error2))

        con.commit()
        con.close()

        con = sqlite3.connect(self.database_path)
        cur = con.cursor()
        cur.execute('ATTACH DATABASE ? AS local',
                    (self.analysis_database_path,))
        cur.execute('''
            INSERT INTO local.correspondence(sequence, detector_type, feature_type,
                feature_count, descriptor_size, descriptor_compressed_size, extraction_time,
                matches_count, matching_time)
            SELECT sequence.name, detector_type.name, feature_type.name,
                F.feature_count, F.descriptor_size, F.descriptor_compressed_size, F.extraction_time,
                M.matches_count, M.matching_time
            FROM
                (
                SELECT image.sequence, image_feature.detector_type, image_feature.feature_type,
                    sum(image_feature.count) AS feature_count,
                    sum(image_feature.size) AS descriptor_size,
                    sum(length(image_feature.data)) AS descriptor_compressed_size,
                    sum(image_feature.time) AS extraction_time
                FROM
                    image_feature
                        INNER JOIN
                    image on image_feature.image = image.id
                GROUP BY image.sequence, image_feature.detector_type, image_feature.feature_type
                ) AS F
                    INNER JOIN
                (
                SELECT image.sequence, image_feature.detector_type, image_feature.feature_type,
                    sum(feature_match.count) AS matches_count,
                    sum(feature_match.time) AS matching_time
                FROM
                    feature_match
                        INNER JOIN
                    image_feature on feature_match.feature1 = image_feature.id
                        INNER JOIN
                    image on image_feature.image = image.id
                GROUP BY image.sequence, image_feature.detector_type, image_feature.feature_type
                ) AS M on F.sequence = M.sequence AND F.feature_type = M.feature_type
                    INNER JOIN
                sequence on M.sequence = sequence.id
                    INNER JOIN
                feature_type as detector_type on M.detector_type = detector_type.id
                    INNER JOIN
                feature_type on M.feature_type = feature_type.id;
            ''')
        cur.execute('''
            INSERT INTO local.reconstruction(sequence, detector_type, feature_type,
                points, observations, error)
            SELECT sequence.name, detector_type.name, feature_type.name, points, observations, error
            FROM
                analysis
                    INNER JOIN
                sequence on analysis.sequence = sequence.id
                    INNER JOIN
                feature_type as detector_type on analysis.detector_type = detector_type.id
                    INNER JOIN
                feature_type on analysis.feature_type = feature_type.id;
            ''')
        con.commit()
        con.close()
