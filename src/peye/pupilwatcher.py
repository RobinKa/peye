from .pupil import Pupil
from .pupilcorrelator import PupilCorrelator

class PupilWatcher:
    def __init__(self, smoothing_factor=0.5, initial_remove_counter=3, max_remove_counter=15, max_distance=None):
        self.pupils = []
        self._correlator = PupilCorrelator(max_distance=max_distance)
        self.smoothing_factor = smoothing_factor
        self.initial_remove_counter = initial_remove_counter
        self.max_remove_counter = max_remove_counter
        self.pupil_remove_counter = {}

    def update(self, locations):
        matches = self._correlator.get_matches(self.pupils, locations)

        removed_pupils = []

        for i, loc in enumerate(matches):
            if i < len(self.pupils):
                pupil = self.pupils[i]
                if loc is not None:
                    # Matched a pupil to a location
                    pupil.update_location(loc)
                    self.pupil_remove_counter[pupil] = min(self.max_remove_counter, self.pupil_remove_counter[pupil] + 1)
                    pupil.certainty = self.pupil_remove_counter[pupil] / self.max_remove_counter
                else:
                    # Removed pupil
                    removed_pupils.append(pupil)
            else:
                # New locations
                pupil = Pupil(loc, self.smoothing_factor)
                self.pupils.append(pupil)
                self.pupil_remove_counter[pupil] = self.initial_remove_counter
                pupil.certainty = self.pupil_remove_counter[pupil] / self.max_remove_counter

        for removed_pupil in removed_pupils:
            self.pupil_remove_counter[removed_pupil] -= 1

            if self.pupil_remove_counter[removed_pupil] <= 0:
                self.pupils.remove(removed_pupil)
                del self.pupil_remove_counter[removed_pupil]