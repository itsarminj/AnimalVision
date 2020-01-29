from vision_model import Vision
import numpy as np

model = Vision('test_img')
model.kernel_size = (10,10)
model.resize_percent = 200
model.real_time(model.human, model.dog_vision, model.fisheye)


