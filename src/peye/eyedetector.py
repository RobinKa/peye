import cv2

class EyeDetector:
    def __init__(self, cascade_filename, max_eyes=None, inset=0, downscales=0):
        self.eyes_cascade = cv2.CascadeClassifier(cascade_filename)
        self.max_eyes = max_eyes
        self.inset = inset
        self.downscales = downscales

    def detect(self, im):
        detected_eyes = []

        scale_factor = 2**self.downscales
        for i in range(self.downscales):
            im = cv2.pyrDown(im)

        eyes = self.eyes_cascade.detectMultiScale(
            im,
            minSize = (self.inset//scale_factor, self.inset//scale_factor),
        )
        
        for (ex, ey, ew, eh) in eyes:
            detected_eyes.append((scale_factor*ex + self.inset,
                                  scale_factor*ey + self.inset,
                                  scale_factor*(ex + ew) - self.inset,
                                  scale_factor*(ey + eh) - self.inset))

        if self.max_eyes:
            detected_eyes = sorted(detected_eyes, key=lambda e: -(e[2] - e[0]) * (e[3] - e[1]))[:self.max_eyes]
        
        return detected_eyes
