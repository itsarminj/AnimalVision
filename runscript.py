from vision_model import Vision
import numpy as np

model = Vision('test_img')
model.load_image()
model.show_image()
print(np.shape(model.img))
