import cv2
import numpy as np

class PupilCorrelator:
    def __init__(self, max_distance=None):
        self.max_distance = max_distance
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def get_matches(self, pupils, locations):
        '''
        Matches the pupils with the locations
        Returns: 
        location_candidate (Same indices as pupils, None: removed pupil, more elements: new locations)

        More locations than pupils:

        Pupils: [a, b, None, None]
        Locations: [u, v, w, x]
        Result: [w, x, u, v]
        Interpretation: a=w, b=x, u and v are new pupil locations

        More pupils than locations:

        Pupils: [a, b, c, d]
        Locations: [u, v, None, None]
        Result: [None, None, u, v]
        Interpretation: c=u, d=v, a and b are removed pupils
        '''

        if len(locations) == 0:
            return [ None ] * len(pupils)

        matches = self.matcher.match(np.array([pupil.location for pupil in pupils], np.float32), np.array(locations, np.float32))
        c = pupils.copy()

        matched_loc_indices = []
        matched_pup_indices = []

        # Match pupil<->location
        for match in matches:
            if self.max_distance is None or match.distance <= self.max_distance:
                c[match.queryIdx] = locations[match.trainIdx]
                matched_loc_indices.append(match.trainIdx)
                matched_pup_indices.append(match.queryIdx)

        # New locations
        for i, loc in enumerate(locations):
            if not i in matched_loc_indices:
                c.append(loc)

        # Removed pupils
        for i in range(len(pupils)):
            if not i in matched_pup_indices:
                c[i] = None
        
        return c
