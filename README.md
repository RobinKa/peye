# peye
A python library to quickly and accurately localize the eyes' pupils
[Demo video](https://www.youtube.com/watch?v=zMMMuSPQkhk)

# Dependencies
- Python 3
- Python libraries
  - numpy
  - cv2
  - [pytocl](https://github.com/ToraxXx/pytocl) (optional for OpenCL support)
  - sklearn (optional for clustering)
  
# Usage
[Webcam example](https://github.com/ToraxXx/peye/blob/master/examples/camera.py)

```python
import cv2
from peye import EyeDetector, PupilDetector

# Load some image
image = cv2.imread("someimage.png")

# Detect the bounding boxes of eyes on the image
eye_detector = EyeDetector()
eyes = eye_detector.detect(image)

# Detect the pupils' coordinates using the eyes' bounding boxes
# Specify the maximum amount of pixels to be considered without downscaling
# (Higher = more accurate but slower)
# Other parameters: 
# cluster_mode=None/"bestpixel"/"bestcluster"
# use_opencl=False/True
# opencl_context=None (None will use cl.create_some_context(False))
pupil_detector = PupilDetector(500)
pupils = [pupil_detector.detect(eye[eye[1]:eye[3], eye[0]:eye[2]]) for eye in eyes]
```

# OpenCL support (requires pytocl)
PupilDetector supports OpenCL by passing `use_opencl=True` and optionally a `pyopencl.Context` as `opencl_context`.
For 4096 maximum pixels to consider the speedup is about 5x-10x for me on a GPU.
For 1024 maximum pixels the speed is about the same.
The current implementation uses a lot of memory and is also not very optimized.

# Clustering
Instead of just outputting the single best pixel it is also possible to let the algorithm cluster the objective function and output
either the center of the cluster of the single best pixel or the center of the cluster with the best average objective by
passing `cluster_mode="bestpixel"` or `cluster_mode="bestcluster"` respectively.
This might yield more accurate results at the cost of some additional runtime.

# Contributors
- Toraxxx (Developer)
