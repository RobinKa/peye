import math
import cv2
import numpy as np
from pytocl import *

def _make_cl_func(max_pixel_count):
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
        
    global_size = (max_pixel_count, max_pixel_count)

    ad_image = CLArgDesc(CLArgType.float32_array, array_size=max_pixel_count**2)
    ad_gradients = CLArgDesc(CLArgType.float32_array, array_size=2*max_pixel_count**2)
    ad_output = CLArgDesc(CLArgType.float32_array, array_size=max_pixel_count**2)
    ad_rows = CLArgDesc(CLArgType.int32)
    ad_cols = CLArgDesc(CLArgType.int32)
    
    func_desc = (CLFuncDesc(calc_objective, global_size)
                .arg(ad_image).copy_in()
                .arg(ad_gradients).copy_in()
                .arg(ad_output, is_readonly=False).copy_out()
                .arg(ad_rows).copy_in()
                .arg(ad_cols).copy_in())
                
    cl_func = CLFunc(func_desc).compile()
    
    def run(image, gradients, objectives):
        if(2 * len(objectives) != len(gradients.flatten()) or len(image.flatten()) != len(objectives)):
            raise Exception("Invalid size for inputs")
        
        cl_func({
            ad_image: image.flatten(),
            ad_gradients: gradients.flatten(),
            ad_output: objectives,
            ad_rows: image.shape[0],
            ad_cols: image.shape[1],
        })
        
    return run

class PupilDetector:
    def __init__(self, max_pixel_count):
        self.max_pixel_count = max_pixel_count
        self.calc_objective = _make_cl_func(max_pixel_count)

    def detect(self, image):
        if image.shape[0] <= 1 or image.shape[1] <= 1:
            return (0, 0)
    
        image_array = np.array(image, np.float32)
    
        pixels = image_array.shape[0] * image_array.shape[1]
        pyr_count = max(0, math.ceil(math.log(pixels / self.max_pixel_count, 4)))
        
        for i in range(pyr_count):
            image_array = cv2.pyrDown(image_array)
        
        image_array /= 255.0
        pixels = image_array.shape[0] * image_array.shape[1]

        image_grad = np.array(np.gradient(image_array), np.float32)
        
        g = np.sqrt(image_grad[0, :, :] * image_grad[0, :, :] + image_grad[1, :, :] * image_grad[1, :, :])
        
        # Calculate the gradients mean and stddev for thresholding it
        grad_mean = np.mean(g)
        grad_stddev = np.std(g)
        grad_threshold = 0.3 * grad_stddev + grad_mean

        image_grad = (g > grad_threshold) * image_grad / (0.0001 + g)
        
        # Calculate the objective
        objectives = np.zeros(image_array.shape, np.float32).flatten()

        self.calc_objective(image_array, image_grad, objectives)
        
        objectives = objectives.reshape(image_array.shape)
        (highest_i, highest_j) = np.unravel_index(objectives.argmax(), image_array.shape)
        
        return ((2**pyr_count)*highest_i, (2**pyr_count)*highest_j)
