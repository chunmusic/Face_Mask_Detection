# Face Mask Detection

- This repository is created for differentiating people who are wearing a mask, not wearing a mask and wearing a mask incorrectly.

- The model was trained based on SSD Inception V2 architecture by Tensorflow 1.

- It can run up to 30 FPS on RTX3080 and 7-9 FPS on Nvidia Jetson NX with CUDA Backend.

- It can run up to 10 FPS on i7-8550u

- The dataset used for training this model is from (https://www.kaggle.com/andrewmvd/face-mask-detection)

## How to use

- Please run the following command to use your webcam as a detection camera.

```python
python video_nms.py
```

## Preview

### With mask

![Image1](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/with_mask.png)


### Without mask

![Image2](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/without_mask.png)

### Wear mask incorrect

![Image3](https://raw.githubusercontent.com/chunmusic/Face_Mask_Detection/master/screenshot/wear_mask_incorrect.png)
