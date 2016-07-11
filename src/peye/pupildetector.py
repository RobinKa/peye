import math
import itertools
import cv2
import numpy as np
import importlib
from os import cpu_count
from multiprocessing.pool import Pool

# Optional pytocl for opencl
if importlib.util.find_spec("pytocl"):
    from pytocl import *
    import pyopencl as cl

# Optional sklearn for clustering
if importlib.util.find_spec("sklearn"):
    from sklearn.cluster import KMeans

def _make_cl_func(context, max_pixel_count):
    def calc_objective(image, nonzero_grads, nonzero_coords, nonzero_count, output, rows, cols):
        i = get_global_id(0)
        j = get_global_id(1)

        if i >= rows or j >= cols:
            return
        
        objective = 0
        
        for i_nz_index in range(nonzero_count):
            g_i = nonzero_coords[2*i_nz_index]
            g_j = nonzero_coords[2*i_nz_index+1]

            d_i = g_i - i
            d_j = g_j - j
            
            mag = sqrt(d_i * d_i + d_j * d_j)
                
            d_i /= mag + 0.001
            d_j /= mag + 0.001
                
            dot = max(0.0, d_i * nonzero_grads[i_nz_index] + d_j * nonzero_grads[i_nz_index + nonzero_count])
                
            objective += dot * dot

        objective *= 1.0 - image[j + i * cols]
        output[j + i * cols] = objective
        
    global_size = (max_pixel_count, max_pixel_count)

    ad_image = CLArgDesc(CLArgType.float32_array, array_size=max_pixel_count**2)
    ad_nonzero_grads = CLArgDesc(CLArgType.float32_array, array_size=2*max_pixel_count**2) # [nz_index] -> (gx(x, y), gy(x, y))
    ad_nonzero_coords = CLArgDesc(CLArgType.int32_array, array_size=2*max_pixel_count**2) # [nz_index] -> (x, y)
    ad_nonzero_count = CLArgDesc(CLArgType.int32)
    ad_output = CLArgDesc(CLArgType.float32_array, array_size=max_pixel_count**2)
    ad_rows = CLArgDesc(CLArgType.int32)
    ad_cols = CLArgDesc(CLArgType.int32)
    
    func_desc = (CLFuncDesc(calc_objective, global_size)
                .arg(ad_image).copy_in()
                .arg(ad_nonzero_grads).copy_in()
                .arg(ad_nonzero_coords).copy_in()
                .arg(ad_nonzero_count).copy_in()
                .arg(ad_output, is_readonly=False).copy_out()
                .arg(ad_rows).copy_in()
                .arg(ad_cols).copy_in())
    
    cl_func = CLFunc(func_desc).compile(context or cl.create_some_context(False))
    
    def run(image, gradients, nonzeros, objectives):
        if(2 * len(objectives) != len(gradients.flatten()) or len(image.flatten()) != len(objectives)):
            raise Exception("Invalid size for inputs")
        
        nonzero_coords = np.array(nonzeros, np.int32).T
        nonzero_count = nonzero_coords.shape[0]
        nonzero_grads = gradients[:, nonzero_coords[:, 0], nonzero_coords[:, 1]]

        cl_func({
            ad_image: image.flatten(),
            ad_nonzero_grads: nonzero_grads.flatten(),
            ad_nonzero_coords: nonzero_coords.flatten(),
            ad_nonzero_count: nonzero_count,
            ad_output: objectives,
            ad_rows: image.shape[0],
            ad_cols: image.shape[1],
        })
        
    return run

def _get_objective(params):
    '''
    0: image_coords
    1: gradients
    2: nz
    '''

    image_coords = params[0]
    gradients = params[1]
    nz = params[2]

    # Calculate the normalized displacements from all image-coords to the gradient coord
    # ((,2) - (p, 2)) / (p, 1) = (p, 2)
    d = nz - image_coords
    disp = d / (0.001 + np.linalg.norm(d, axis=1).reshape(-1, 1))

    # Get the gradient at the gradient coord
    # (2,)
    grad = gradients[:, nz[0], nz[1]]

    # Calculate the dot product between the normalized displacements and the gradient and square it
    # (p, 2) * (2,) = (p,)
    return np.inner(disp, grad)**2

def _get_objectives(image_array, gradients, nonzeros, pool):
    # Produce all possible image and nonzero coordinates
    image_coords = np.array(list(itertools.product(range(image_array.shape[0]), range(image_array.shape[1]))))
    nonzero_coords = np.array(nonzeros, np.int32).T

    # Use a thread pool to calculate all the objective values and sum them up after
    params = [(image_coords, gradients, nz) for nz in nonzero_coords]
    objectives = np.sum(pool.map(_get_objective, params), axis=0)

    objectives = np.maximum(objectives, 0) * (1 - image_array.flatten())
    
    return objectives

class PupilDetector:
    def __init__(self, max_pixel_count, cluster_mode=None, use_opencl=False, opencl_context=None):
        self.max_pixel_count = max_pixel_count
        self.cluster_mode = cluster_mode
        self.use_opencl=use_opencl

        if cluster_mode == "bestcluster":
            self.cluster = KMeans(n_clusters=20, n_init=1, tol=0.01, max_iter=30, precompute_distances=True, verbose=0)
        elif cluster_mode == "bestpixel":
            self.cluster = KMeans(n_clusters=10, n_init=5, tol=0.01, max_iter=30, precompute_distances=True, verbose=0)
        elif cluster_mode != None:
            raise Exception("Unknown cluster mode " + str(cluster_mode))

        if use_opencl:
            self.calc_objective = _make_cl_func(opencl_context, max_pixel_count)
        else:
            cpus = cpu_count()
            if cpus is None:
                print("Warning: Could not detect cpu count with os.cpu_count(), defaulting to 1")
                cpus = 1
            self.pool = Pool(cpus)

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

        # Calculate the objective using opencl or normal numpy operations
        objectives = None
        if self.use_opencl:
            objectives = np.zeros(image_array.shape, np.float32).flatten()
            self.calc_objective(image_array, image_grad, (g > grad_threshold).nonzero(), objectives)
        else:
            objectives = _get_objectives(image_array, image_grad, (g > grad_threshold).nonzero(), self.pool)

        objectives = objectives.reshape(image_array.shape)

        (highest_i, highest_j) = np.unravel_index(objectives.argmax(), image_array.shape)
        
        if self.cluster_mode != None:
            # Normalize the objective to be between 0 and 1 for clustering
            min_objective = np.min(objectives)
            max_objective = np.max(objectives)
            objectives = (objectives - min_objective) / (max_objective - min_objective)
            data = []

            for i in range(image_array.shape[0]):
                for j in range(image_array.shape[1]):
                    data.append((i / image_array.shape[0], j / image_array.shape[1], objectives[i, j]))

            # data[b, f]
            # f: (i, j, objective)
            data = np.array(data)
            
            cluster_result = self.cluster.fit(data)

            if self.cluster_mode == "bestpixel":
                # Get the cluster center of the pixel with the highest objective

                highest_center = cluster_result.cluster_centers_[cluster_result.labels_[image_array.shape[1] * highest_i + highest_j]]
                highest_i = int(round(highest_center[0] * image_array.shape[0]))
                highest_j = int(round(highest_center[1] * image_array.shape[1]))
            elif self.cluster_mode == "bestcluster":
                # Get the cluster center with the highest average objective

                scores = np.zeros(self.cluster.n_clusters)
                label_count = np.zeros(self.cluster.n_clusters)
                for index, label in enumerate(cluster_result.labels_):
                    obj_index = np.unravel_index(index, objectives.shape)
                    scores[label] += objectives[obj_index[0], obj_index[1]]
                    label_count[label] += 1

                for label, label_count in enumerate(label_count):
                    scores[label] /= label_count

                highest_center = cluster_result.cluster_centers_[np.argmax(scores)]
                highest_i = int(round(highest_center[0] * image_array.shape[0]))
                highest_j = int(round(highest_center[1] * image_array.shape[1]))

        return ((2**pyr_count)*highest_i, (2**pyr_count)*highest_j)
