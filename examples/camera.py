import cv2
from peye import EyeDetector, PupilDetector, PupilWatcher
from time import time
import numpy as np

MAX_PIXEL_COUNT = 80*80
IMAGE_PATH = "image_%s.png"
        
def main():
    image_count = 0
    
    eye_detector = EyeDetector("haarcascade_eye_tree_eyeglasses.xml", max_eyes=2, inset=15, downscales=0)
    pupil_detector = PupilDetector(MAX_PIXEL_COUNT, use_opencl=True)
    pupil_watcher = PupilWatcher(smoothing_factor=0.3, initial_remove_counter=1, max_remove_counter=15, max_distance=None)
    
    cap = cv2.VideoCapture(0)

    fps_total = 0
    fps_eyes = 0
    fps_pupils = 0
    fps_watcher = 0

    while True:
        t_total = time()
        ret, frame = cap.read()

        if not ret:
            break

        #frame = cv2.blur(cv2.pyrDown(cv2.transpose(frame)[200:-200, :]), (3, 3))
        frame = cv2.blur(frame, (3, 3))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t_eyes = time()
        eyes = eye_detector.detect(gray)
        fps_eyes = 0.5 * fps_eyes + 0.5 / max(0.001, time() - t_eyes)

        #for e in eyes:
        #    cv2.rectangle(frame, (e[0], e[1]), (e[2], e[3]), (255, 0, 0), 2)

        t_pupils = time()
        pupils = [pupil_detector.detect(gray[eye[1]:eye[3], eye[0]:eye[2]]) for eye in eyes]
        fps_pupils = 0.5 * fps_pupils + 0.5 / max(0.001, time() - t_pupils)

        detected_pupil_locations = [np.array([eye[0] + pupil[0], eye[1] + pupil[1]]) for (eye, pupil) in zip(eyes, pupils)]

        # Draw detected locations
        for pupil_location in detected_pupil_locations:
            #cv2.circle(frame, (int(round(pupil_location[0])), int(round(pupil_location[1]))), 3, (0, 255, 0), 1)
            pass

        t_watcher = time()
        # Draw watched locations
        pupil_watcher.update(detected_pupil_locations)
        fps_watcher = 0.5 * fps_watcher + 0.5 / max(0.001, time() - t_watcher)
        for pupil in filter(lambda p: p.certainty >= 0.7, pupil_watcher.pupils):
            cv2.circle(frame, (int(round(pupil.location[0])), int(round(pupil.location[1]))), 3, (0, 0, 255), 1)
            cv2.putText(frame, str(pupil.id), (int(round(pupil.location[0])), int(round(pupil.location[1] + 20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.putText(frame, "FPS: %.1f" % fps_total, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255))
        cv2.putText(frame, "FPS Eyes: %.1f" % fps_eyes, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255))
        cv2.putText(frame, "FPS Pupils: %.1f" % fps_pupils, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255))
        cv2.putText(frame, "FPS Watcher: %.1f" % fps_watcher, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255))

        cv2.imshow("frame", frame)
        
        if(len(eyes) > 0):
            #cv2.imwrite(IMAGE_PATH % image_count, frame)
            image_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps_total = 0.5 * fps_total + 0.5 / max(0.001, time() - t_total)
        print("FPS Total, Eyes, Pupils, Watcher:", "%.2f" % fps_total, "%.2f" % fps_eyes, "%.2f" % fps_pupils, "%.2f" % fps_watcher)
        print("T Eyes, Pupils, Watcher:", "%.2f" % (fps_total / fps_eyes), "%.2f" % (fps_total / fps_pupils), "%2.f" % (fps_total / fps_watcher))

    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
        