import cv2
from peye import EyeDetector, PupilDetector
from time import time

MAX_PIXEL_COUNT = 64 * 64
IMAGE_PATH = "image_%s.png"
        
def main():
    image_count = 0
    
    eye_detector = EyeDetector("haarcascade_eye.xml")
    pupil_detector = PupilDetector(MAX_PIXEL_COUNT, cluster_mode=None, use_opencl=False)
    
    cap = cv2.VideoCapture(0)

    fps_total = 0
    fps_eyes = 0
    fps_pupils = 0

    while True:
        t_total = time()
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t_eyes = time()
        eyes = eye_detector.detect(gray)
        fps_eyes = 0.5 * fps_eyes + 0.5 / max(0.001, time() - t_eyes)
        print("FPS Eyes:", fps_eyes)

        t_pupils = time()
        pupils = [pupil_detector.detect(gray[eye[1]:eye[3], eye[0]:eye[2]]) for eye in eyes]
        fps_pupils = 0.5 * fps_pupils + 0.5 / max(0.001, time() - t_pupils)
        print("FPS Pupils:", fps_pupils)

        for eye, pupil in zip(eyes, pupils):
            cv2.rectangle(
                gray, 
                (eye[0] + pupil[0] - 2, eye[1] + pupil[1] - 2), 
                (eye[0] + pupil[0] + 2, eye[1] + pupil[1] + 2), 
                (255, 0, 0), 
                2
            )
        
        cv2.imshow("frame", gray)
        
        if(len(eyes) > 0):
            #cv2.imwrite(IMAGE_PATH % image_count, gray)
            image_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        fps_total = 0.5 * fps_total + 0.5 / max(0.001, time() - t_total)
        print("FPS Total, Eyes, Pupils:", "%.2f" % fps_total, "%.2f" % fps_eyes, "%.2f" % fps_pupils)
        print("T Eyes, %Pupils:", "%.2f" % (fps_total / fps_eyes), "%.2f" % (fps_total / fps_pupils))
    
if __name__ == "__main__":
    main()
        