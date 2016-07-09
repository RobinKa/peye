from distutils.core import setup

setup(
    name="peye",
    version= "0.1.0",
    description="A python library to quickly and accurately localize the eyes' pupils",
    author="Robin Kahlow (Toraxxx)",
    author_email="xtremegosugaming@gmail.com",
    maintainer="Robin Kahlow (Toraxxx)",
    maintainer_email="xtremegosugaming@gmail.com",
    url="https://github.com/ToraxXx/peye",
    requires=["numpy", "cv2", "pytocl"],
    license= "MIT",
    package_dir={"": "src"},
    packages=["peye"],
)
