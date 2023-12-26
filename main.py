from os import listdir
import pandas as pd
import  numpy as np
from matplotlib.pyplot import figure, imshow, axis
import matplotlib.pyplot as plt
from PIL import Image

for i in range(10):
    image = np.random.rand(3,128,128)
    filename0 = fr"C:\Users\ali\PycharmProjects\ACGAN-Kashi\generated\{i}.jpg"
    im1 = Image.fromarray(np.array(image.T * 255, dtype='uint8'))
    im1 = im1.save(filename0)



