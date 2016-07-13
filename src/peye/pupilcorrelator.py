import itertools

class PupilCorrelator:
    def _get_error(pupils, locations):
        error = 0
        
        for pupil, loc in zip(pupils, locations):
            if pupil is not None and loc is not None:
                dx = pupil.location[0] - loc[0]
                dy = pupil.location[1] - loc[1]
                error += dx*dx + dy*dy

        return error

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

        matched_pupils = []
        new_locations = []
        removed_pupils = []

        # More pupils than locations: remove worst pupils
        # More locations than pupils: add new pupils
        # Same: match

        location_candidates = locations.copy()
        pupil_candidate = pupils.copy()

        len_diff = len(locations) - len(pupils)

        if(len_diff < 0):
            location_candidates += [None] * -len_diff
        else:
            pupil_candidate += [None] * len_diff

        location_candidates = itertools.permutations(location_candidates)
        
        # Find the location assignments that cause the lowest error
        min_error = None
        min_location_candidate = None
        for location_candidate in location_candidates:
            error = PupilCorrelator._get_error(pupil_candidate, location_candidate)
            if min_error is None or error < min_error:
                min_location_candidate = location_candidate
                min_error = error
        
        return min_location_candidate