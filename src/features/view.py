import cv2
import numpy as np


class View:
    def __init__(self, idx, image_id, image_path):
        self.idx = idx
        self.image_id = image_id
        self.image_path = image_path
        self.image = None
        self.feature_id = None
        self.keypoints = self.descriptors = None
        self.Rt = self.R = self.t = None
        self.pnts_3d_idx = np.empty((0, 1))

    def load(self):
        self.image = cv2.imread(self.image_path)

    def setFeature(self, feature):
        self.keypoints, self.descriptors = feature

    def setFeatureID(self, feature_id):
        self.feature_id = feature_id

    def setRt(self, R, t, ref_Rt=None):
        self.Rt = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        if ref_Rt is not None:
            self.Rt = self.Rt @ ref_Rt
        self.R = self.Rt[:3, :3]
        self.t = self.Rt[:3, [3]]

    def setPnts_3d_idx(self, idxs, n):
        for idx in idxs:
            if self.pnts_3d_idx[idx.astype(int)] == -1:
                self.pnts_3d_idx[idx.astype(int)] = n
            n += 1

    def projection(self, K):
        return K @ self.Rt[:3, :]

    def reproject(self, K, pts_3d, pts):
        pts_prj = cv2.convertPointsFromHomogeneous(
            (K @ (self.R @ pts_3d.T + self.t)).T
        )[:, 0, :]
        error = np.sqrt(np.sum((pts_prj - pts) ** 2, axis=1))[np.newaxis].T
        return pts_prj, error
