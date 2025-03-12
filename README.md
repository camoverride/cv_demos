# Computer Vision Demos for Seattle Makers


## Setup

Create a virtual environment:
- `python3 -m venv .venv`
- `source .venv/bin/activate`

Install the requirements. NOTE: installing mediapipe automatically installs opencv
- `pip install mediapipe`

Download required models:

```
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

curl -O https://pjreddie.com/media/files/yolov3.weights
curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```


## Run

`python face_detection_demo.py`
