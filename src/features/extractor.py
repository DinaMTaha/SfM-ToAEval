import cv2
import logging


class Extractor:
    def __init__(self, detector_type, feature_type):
        self.feature_type = feature_type
        self.detector_type = detector_type
        feature_algorithm = {
            'SIFT': cv2.xfeatures2d.SIFT_create(),
            'SURF': cv2.xfeatures2d.SURF_create(extended=True),
            'DAISY': cv2.xfeatures2d.DAISY_create(use_orientation=True),
            'BRISK': cv2.BRISK_create(),
            'KAZE': cv2.KAZE_create(extended=True),
            'AKAZE': cv2.AKAZE_create(descriptor_size=0, descriptor_channels=3),
            'ORB': cv2.ORB_create(nfeatures=20000),
            'FREAK': cv2.xfeatures2d.FREAK_create(),
            'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=64, use_orientation=True),
            'LUCID': cv2.xfeatures2d.LUCID_create(lucid_kernel=15, blur_kernel=2),
            'LATCH': cv2.xfeatures2d.LATCH_create(bytes=64, rotationInvariance=True),
            'BEBLID': cv2.xfeatures2d.BEBLID_create(n_bits=101, scale_factor=1 if self.detector_type == 'ORB' else
                                                    6.75 if self.detector_type == 'SIFT' else
                                                    6.25 if self.detector_type in ['SURF', 'KAZE'] else
                                                    5.00),
            'TEBLID': cv2.xfeatures2d.TEBLID_create(n_bits=103, scale_factor=1 if self.detector_type == 'ORB' else
                                                    6.75 if self.detector_type == 'SIFT' else
                                                    6.25 if self.detector_type in ['SURF', 'KAZE'] else
                                                    5.00),
            'VGG': cv2.xfeatures2d.VGG_create(desc=100, scale_factor=0.75 if self.detector_type == 'ORB' else
                                              6.75 if self.detector_type == 'SIFT' else
                                              6.25 if self.detector_type in ['SURF', 'KAZE'] else
                                              5.00),
            'FAST': cv2.FastFeatureDetector_create(),
            'AGAST': cv2.AgastFeatureDetector_create(),
            'STAR': cv2.xfeatures2d.StarDetector_create(maxSize=128, responseThreshold=10),
        }
        if self.feature_type in feature_algorithm:
            if self.detector_type in feature_algorithm:
                self.detector = feature_algorithm[self.detector_type]
                self.descriptor = feature_algorithm[self.feature_type]
            elif self.detector_type is None:
                self.detector = None
                self.detector_type = self.feature_type
                self.descriptor = feature_algorithm[self.feature_type]
                try:
                    self.descriptor.detectAndCompute(None, None)
                except Exception as e:
                    if e.code == cv2.Error.StsNotImplemented:
                        self.detector_type = 'FAST'
                        self.detector = feature_algorithm[self.detector_type]
                    else:
                        pass
            else:
                logging.error(f'Invalid feature type <{self.detector_type}>!')
                raise Exception(
                    f'Invalid feature type <{self.detector_type}>!')
        else:
            logging.error(f'Invalid feature type <{self.feature_type}>!')
            raise Exception(f'Invalid feature type <{self.feature_type}>!')

    def extract(self, view):
        if self.detector:
            keypoints = self.detector.detect(view.image, None)
            if self.feature_type in ['AKAZE', 'KAZE']:
                min_class_id = min(
                    [keypoint.class_id for keypoint in keypoints])
                for keypoint in keypoints:
                    keypoint.class_id += 1 - min_class_id
            elif self.feature_type in ['DAISY']:
                for keypoint in keypoints:
                    keypoint.angle = max(0, keypoint.angle)
            keypoints, descriptors = self.descriptor.compute(
                view.image, keypoints)
        else:
            keypoints, descriptors = self.descriptor.detectAndCompute(
                view.image, None)
        return (keypoints, descriptors)
