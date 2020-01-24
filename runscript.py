from vision_model import Vision
import numpy as np

model = Vision('test_img')
model.real_time(model.human, model.dog_vision, model.snake_vision)

