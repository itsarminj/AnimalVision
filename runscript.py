from vision_model import Vision
import numpy as np

model = Vision('test_img')
model.real_time()
print(np.shape(model.img))
