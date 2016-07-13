from .pupil import Pupil
from .pupilcorrelator import PupilCorrelator

class PupilWatcher:
    def __init__(self, smoothing_factor=0.5, remove_delay=10):
        self.pupils = []
        self._correlator = PupilCorrelator()
        self.smoothing_factor = smoothing_factor
        self.remove_delay = remove_delay

    def update(self, locations):
        matches = self._correlator.get_matches(self.pupils, locations)

        removed_pupils = []

        for i, loc in enumerate(matches):
            if i < len(self.pupils):
                if loc is not None:
                    # Matched a pupil to a location
                    self.pupils[i].update_location(loc)
                    self.pupils[i].remove_count = 0
                else:
                    # Removed pupil
                    removed_pupils.append(self.pupils[i])
            else:
                # New locations
                self.pupils.append(Pupil(loc, self.smoothing_factor))

        for removed_pupil in removed_pupils:
            removed_pupil.remove_count += 1
            if removed_pupil.remove_count > self.remove_delay:
                self.pupils.remove(removed_pupil)