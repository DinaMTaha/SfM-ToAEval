import glob
import time
import os
import numpy as np
from .features.extractor import Extractor
from .features.matcher import Matcher
from .features.view import View
from .colmap import database as cmdb


class Experiment:
    def __init__(self, K, database, sequence, root_path,
                 image_format='jpg', detector_type='SIFT', feature_type='SIFT',
                 max_features=2000, max_matches=500):
        self.K = K
        self.database = database
        self.sequence_id = database.getSequenceId(sequence)
        self.feature_type = feature_type
        self.detector_type_id = database.getFeatureTypeId(detector_type)
        self.feature_type_id = database.getFeatureTypeId(feature_type)
        self.root_path = root_path
        self.image_format = image_format
        self.extractor = Extractor(detector_type, feature_type)
        self.matcher = Matcher(
            self.extractor.descriptor.defaultNorm(), crossCheck=True)
        self.max_features = max_features
        self.max_matches = max_matches
        self.views = []
        self.matches = {}

    def createViews(self):
        self.views = []
        files = sorted(glob.glob(os.path.join(
            self.root_path, '*.' + self.image_format)))
        keypoints_count = 0
        for idx, file in enumerate(files):
            file = os.path.abspath(file)
            view = View(idx, self.database.getImageId(
                file, self.sequence_id), os.path.join(self.root_path, file))
            view.load()
            feature = self.database.getFeature(
                view.image_id, self.detector_type_id, self.feature_type_id)
            if feature:
                view.setFeatureID(feature[0])
                view.setFeature(feature[1])
                keypoints_count += len(feature[1][0])
            else:
                tic2 = time.time()
                keypoints, descriptors = self.extractor.extract(view)
                keypoints_count += len(keypoints)
                toc2 = time.time()
                sort_order = np.argsort(
                    [-keypoint.response for keypoint in keypoints])
                sort_order = sort_order[:min(
                    self.max_features, len(sort_order))]
                view.keypoints = [AttrDict({
                    'pt': keypoints[i].pt,
                    'size':keypoints[i].size,
                    'angle':keypoints[i].angle,
                    'class_id':keypoints[i].class_id,
                    'octave':keypoints[i].octave,
                    'response':keypoints[i].response
                }) for i in sort_order]
                view.descriptors = np.array(
                    [descriptors[i] for i in sort_order])
                view.setFeatureID(
                    self.database.putFeature(view.image_id,
                                             self.detector_type_id,
                                             self.feature_type_id, len(
                                                 keypoints),
                                             view.descriptors.nbytes, toc2 - tic2, (view.keypoints, view.descriptors)))
            self.views.append(view)

    def createMatches(self):
        self.matches = {}
        matches_count = 0
        for i in range(0, len(self.views)-1):
            for j in range(i+1, len(self.views)):
                match = self.database.getMatch(
                    self.views[j].feature_id, self.views[i].feature_id)
                if match:
                    self.matches[(j, i)] = match[1]
                    matches_count += len(match[1])
                else:
                    tic2 = time.time()
                    matches = self.matcher.match(
                        self.views[j], self.views[i])
                    toc2 = time.time()
                    matches_count += len(matches)
                    self.matches[(j, i)] = [AttrDict({
                        'imgIdx': match.imgIdx,
                        'queryIdx': match.queryIdx,
                        'trainIdx': match.trainIdx,
                        'distance': match.distance
                    }) for match in matches[:min(self.max_matches, len(matches))]]
                    self.database.putMatch(self.views[j].feature_id, self.views[i].feature_id,
                                           len(matches), toc2 - tic2, self.matches[(j, i)])

    def toColmap(self, colmap_db_path):
        colmap_db = cmdb.COLMAPDatabase.connect(colmap_db_path)
        colmap_db.create_tables()
        fx, fy, cx, cy, s = (self.K[0, 0], self.K[1, 1],
                             self.K[0, 2], self.K[1, 2],
                             self.K[0, 1])
        camera_id = colmap_db.add_camera(1,
                                         self.views[0].image.shape[1],
                                         self.views[0].image.shape[0],
                                         [fx, fy, cx, cy])
        for view in self.views:
            _, image_name = os.path.split(view.image_path)
            view.colmap_id = colmap_db.add_image(image_name, camera_id)
            colmap_db.add_keypoints(view.colmap_id,
                                    np.array([[keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle]
                                              for keypoint in view.keypoints], dtype=float))
            colmap_db.add_descriptors(view.colmap_id, np.zeros(
                (len(view.keypoints), 128), dtype=int))
        for pair in self.matches:
            colmap_db.add_matches(
                self.views[pair[0]].colmap_id,
                self.views[pair[1]].colmap_id,
                np.array([[match.queryIdx, match.trainIdx] for match in self.matches[pair]], dtype=int))
        colmap_db.commit()
        colmap_db.close()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
