import cv2
import logging


class Matcher:
    def __init__(self, normType, crossCheck=True):
        self.normType = normType
        self.matcher = cv2.BFMatcher(self.normType, crossCheck)

    def match(self, view1, view2):
        if view1 is None:
            matches = self.matcher.match(view2.descriptors)
        else:
            matches = self.matcher.match(view1.descriptors, view2.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            logging.debug('Matched %d features for images\n\t%s and\n\t%s', len(
                matches), view1.image_path, view2.image_path)
        return matches
