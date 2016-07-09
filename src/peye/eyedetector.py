import cv2

class EyeDetector:
    def __init__(self):
        self.eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    def detect(self, im):
        detected_eyes = []

        eyes = self.eyes_cascade.detectMultiScale(
            im,
        )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(im, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            detected_eyes.append((ex, ey, ex + ew, ey + eh))
        
        return detected_eyes
