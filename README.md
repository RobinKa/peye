# peye
A python library to quickly and accurately localize the eyes' pupils
[Demo video](https://www.youtube.com/watch?v=zMMMuSPQkhk)

# Dependencies
- Python 3
- Python libraries
  - numpy
  - cv2
  - [pytocl](https://github.com/ToraxXx/pytocl)
  
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
pupil_detector = PupilDetector(500)
pupils = [pupil_detector.detect(eye[eye[1]:eye[3], eye[0]:eye[2]]) for eye in eyes]
```

# Contributors
- Toraxxx (Developer)
