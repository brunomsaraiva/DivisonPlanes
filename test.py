import numpy as np
import os
from anglemanagerv2 import AngleManager
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte, img_as_int
from skimage.color import gray2rgb
from tkinter import filedialog
from matplotlib import pyplot as plt

app = AngleManager()
app.load_kymographs()

app.compute_coords(method="PCA")

app.compute_regression()
app.compute_angles()

print(app.kym1_angle, app.kym2_angle, app.angle_diff)

plt.imshow(app.filtered_kym1)
plt.show()
plt.imshow(app.filtered_kym2)
plt.show()
