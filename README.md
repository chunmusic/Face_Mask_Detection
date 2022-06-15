# Face Mask Detection

- This repository is created for differentiating people who are wearing a mask, not wearing a mask and wearing a mask incorrectly.

- The model was trained based on SSD Inception V2 architecture by Tensorflow 1.

- On RTX3080, it can run up to 45 FPS (Python)

- On i7-8550u, it can run up to 10 FPS (Python) and 15 FPS (C++)

- On Nvidia-Jetson NX, it can run up to 7-9 FPS (Python)

- The dataset used for training this model is from (https://www.kaggle.com/andrewmvd/face-mask-detection)

## How to use

#### Normal Operation
- Please run the following command to use your webcam as a detection camera.

```python
python video_nms.py
```
#### Run using Docker
```docker
xhost +local:docker # To allow xhost for docker container
docker compose up --build
```
## Preview

### With mask

![Image1](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/with_mask.png)


### Without mask

![Image2](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/without_mask.png)

### Wear mask incorrect

![Image3](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/wear_mask_incorrect.png)
