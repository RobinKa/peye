import cv2
import numpy as np
import math
import itertools
from pytocl import *
import pyopencl as cl

IMAGE_PATH = "image_%s.png"
image_count = 0
MAX_PIXEL_COUNT = 64*64

def make_cl_func():
    def calc_objective(image, gradients, output, rows, cols):
        i = get_global_id(0)
        j = get_global_id(1)
        
        if i >= rows or j >= cols:
            return
        
        objective = 0
        
        for g_i in range(rows):
            for g_j in range(cols):
                d_i = g_i - i
                d_j = g_j - j
                
                mag = sqrt(d_i * d_i + d_j * d_j)
                
                d_i /= mag + 0.001
                d_j /= mag + 0.001
                
                dot = max(0.0, d_i * gradients[g_j + g_i * cols] + d_j * gradients[g_j + g_i * cols + rows * cols])
                
                objective += dot * dot
            
        objective *= 1.0 - image[j + i * cols]
        output[j + i * cols] = objective
        
    global_size = (MAX_PIXEL_COUNT, MAX_PIXEL_COUNT)

    ad_image = CLArgDesc(CLArgType.float32_array, array_size=MAX_PIXEL_COUNT**2)
    ad_gradients = CLArgDesc(CLArgType.float32_array, array_size=2*MAX_PIXEL_COUNT**2)
    ad_output = CLArgDesc(CLArgType.float32_array, array_size=MAX_PIXEL_COUNT**2)
    ad_rows = CLArgDesc(CLArgType.int32)
    ad_cols = CLArgDesc(CLArgType.int32)
    
    func_desc = (CLFuncDesc(calc_objective, global_size)
                .arg(ad_image).copy_in()
                .arg(ad_gradients).copy_in()
                .arg(ad_output, is_readonly=False).copy_out()
                .arg(ad_rows).copy_in()
                .arg(ad_cols).copy_in())
                
    cl_func = CLFunc(func_desc).compile(cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.CPU)))
    
    def run(image, gradients, objectives):
        assert(2 * len(objectives) == len(gradients.flatten()))
        assert(len(image.flatten()) == len(objectives))
        
        print("Rows:", image.shape[0])
        print("Cols:", image.shape[1])
        print("Pixels:", image.shape[0] * image.shape[1])
    
        cl_func({
            ad_image: image.flatten(),
            ad_gradients: gradients.flatten(),
            ad_output: objectives,
            ad_rows: image.shape[0],
            ad_cols: image.shape[1],
        })
        
    return run

class PupilDetector:
    def __init__(self, pixel_count):
        self.pixel_count = pixel_count
        self.calc_objective = make_cl_func()

    def detect(self, image):
        if image.shape[0] <= 1 or image.shape[1] <= 1:
            return (0, 0)
    
        image_array = np.array(image, np.float32)
    
        pixels = image_array.shape[0] * image_array.shape[1]
        pyr_count = max(0, math.ceil(math.log(pixels / self.pixel_count, 4)))
        print(pixels, pyr_count)
        
        for i in range(pyr_count):
            image_array = cv2.pyrDown(image_array)
        
        image_array /= 255.0
        pixels = image_array.shape[0] * image_array.shape[1]
        print("Pixels:", pixels)
        
        image_grad = np.array(np.gradient(image_array), np.float32)
        
        g = np.sqrt(image_grad[0, :, :] * image_grad[0, :, :] + image_grad[1, :, :] * image_grad[1, :, :])
        
        # Calculate the gradients mean and stddev for thresholding it
        print("Calculating gradient statistics")
        grad_mean = np.mean(g)
        grad_stddev = np.std(g)
        grad_threshold = 0.3 * grad_stddev + grad_mean

        print("Grad threshold:", grad_threshold)
        
        image_grad = (g > grad_threshold) * image_grad / (0.0001 + g)
        
        # Calculate the objective
        objectives = np.zeros(image_array.shape, np.float32).flatten()
        
        print("Calculating objectives")
        self.calc_objective(image_array, image_grad, objectives)
        
        print("Min objective:", np.min(objectives))
        print("Max objective:", np.max(objectives))
        
        objectives = objectives.reshape(image_array.shape)
        (highest_i, highest_j) = np.unravel_index(objectives.argmax(), image_array.shape)
        
        '''
        print("Highest i:", highest_i)
        print("Highest j:", highest_j)
        
        objectives = (objectives - np.min(objectives)) / (np.max(objectives) - np.min(objectives))
        #objectives += (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        
        objectives = cv2.pyrUp(cv2.pyrUp(objectives))
        objectives = cv2.resize(objectives, (256, 256))
        
        cv2.imshow("frame", objectives)
        '''
        
        return ((2**pyr_count)*highest_j, (2**pyr_count)*highest_i)

class EyeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    def detect(self, im):
        # cv2.equalizeHist(im, im)
        
        detected_eyes = []
        
        '''
        faces = self.face_cascade.detectMultiScale(
            im,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags=0
        )

        for (x, y, w, h) in faces:
            roi = im[y:y+h, x:x+w]
        
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            eyes = self.eyes_cascade.detectMultiScale(
                roi,
            )
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(im, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                detected_eyes.append((x + ex, y + ey, x + ex + ew, y + ey + eh))
        '''
        eyes = self.eyes_cascade.detectMultiScale(
            im,
        )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(im, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            detected_eyes.append((ex, ey, ex + ew, ey + eh))
        
        return detected_eyes
   
    
def save_image(img):
    global image_count
    #cv2.imwrite(IMAGE_PATH % image_count, img)
    image_count += 1
        
def main():
    eye_detector = EyeDetector()
    pupil_detector = PupilDetector(MAX_PIXEL_COUNT)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pupils = []
        
        eyes = eye_detector.detect(gray)
        for eye in eyes:
            pupils.append(pupil_detector.detect(gray[eye[1]:eye[3], eye[0]:eye[2]]))
           
        for i, pupil in enumerate(pupils):
            eye = eyes[i]
            cv2.rectangle(
                gray, 
                (eye[0] + pupil[0] - 2, eye[1] + pupil[1] - 2), 
                (eye[0] + pupil[0] + 2, eye[1] + pupil[1] + 2), 
                (255, 0, 0), 
                2
            )
            
            print("Pupil at", eye[0] + pupil[0], eye[1] + pupil[1])
        
        cv2.imshow("frame", gray)
        
        if(len(eyes) > 0):
            save_image(gray)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
if __name__ == "__main__":
    main()
        