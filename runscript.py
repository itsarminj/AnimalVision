from vision_model import Vision
import numpy as np

model = Vision('test_img')
model.kernel_size = (20, 20)
model.resize_percent = 100
args = (model.human, model.fly)
model.real_time(model.human, model.fly, model.dog_vision, model.snake_vision)


