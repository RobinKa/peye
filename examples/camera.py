import cv2
from peye import EyeDetector, PupilDetector, PupilWatcher
from time import time
import numpy as np

MAX_PIXEL_COUNT = 64*64
IMAGE_PATH = "image_%s.png"
        
def main():
    image_count = 0
    
    eye_detector = EyeDetector("haarcascade_eye.xml")
    pupil_detector = PupilDetector(MAX_PIXEL_COUNT, use_opencl=True)
    pupil_watcher = PupilWatcher()

    cap = cv2.VideoCapture(0)

    fps_total = 0
    fps_eyes = 0
    fps_pupils = 0
    fps_watcher = 0

    while True:
        t_total = time()
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t_eyes = time()
        eyes = eye_detector.detect(gray)
        fps_eyes = 0.5 * fps_eyes + 0.5 / max(0.001, time() - t_eyes)

        for e in eyes:
            cv2.rectangle(frame, (e[0], e[1]), (e[2], e[3]), (255, 0, 0), 2)

        t_pupils = time()
        pupils = [pupil_detector.detect(gray[eye[1]:eye[3], eye[0]:eye[2]]) for eye in eyes]
        fps_pupils = 0.5 * fps_pupils + 0.5 / max(0.001, time() - t_pupils)

        detected_pupil_locations = [np.array([eye[0] + pupil[0], eye[1] + pupil[1]]) for (eye, pupil) in zip(eyes, pupils)]

        # Draw detected locations
        for pupil_location in detected_pupil_locations:
            cv2.circle(frame, (int(round(pupil_location[0])), int(round(pupil_location[1]))), 3, (0, 255, 0), 1)

        t_watcher = time()
        # Draw watched locations
        pupil_watcher.update(detected_pupil_locations)
        fps_watcher = 0.5 * fps_watcher + 0.5 / max(0.001, time() - t_watcher)
        for pupil in pupil_watcher.pupils:
            cv2.circle(frame, (int(round(pupil.location[0])), int(round(pupil.location[1]))), 3, (0, 0, 255), 1)
            cv2.putText(frame, str(pupil.id), (int(round(pupil.location[0])), int(round(pupil.location[1] + 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow("frame", frame)
        
        if(len(eyes) > 0):
            #cv2.imwrite(IMAGE_PATH % image_count, gray)
            image_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps_total = 0.5 * fps_total + 0.5 / max(0.001, time() - t_total)
        print("FPS Total, Eyes, Pupils, Watcher:", "%.2f" % fps_total, "%.2f" % fps_eyes, "%.2f" % fps_pupils, "%.2f" % fps_watcher)
        print("T Eyes, Pupils, Watcher:", "%.2f" % (fps_total / fps_eyes), "%.2f" % (fps_total / fps_pupils), "%2.f" % (fps_total / fps_watcher))

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
        