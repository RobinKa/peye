class Pupil:
    next_id = 0

    def __init__(self, loc, smoothing_factor):
        self.smoothing_factor = smoothing_factor
        self.location = loc
        self.id = Pupil.next_id
        self.remove_count = 0
        Pupil.next_id += 1

    def update_location(self, new_loc):
        self.location = self.smoothing_factor * self.location + (1 - self.smoothing_factor) * new_loc
