import numpy as np
import os
from anglemanagerv2 import AngleManager
from skimage.io import imsave
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte, img_as_int
from skimage.color import gray2rgb
from tkinter import filedialog

app = AngleManager()
app.load_kymographs()

app.compute_coords()

app.compute_regression()

save_path = filedialog.askdirectory()

kym1 = img_as_float(gray2rgb(app.kymograph_1))
kym2 = img_as_float(gray2rgb(app.kymograph_2))

kym1_boundaries = mark_boundaries(kym1, img_as_int(app.filtered_kym1), color=(1,0,0))
kym2_boundaries = mark_boundaries(kym2, img_as_int(app.filtered_kym2), color=(1,0,0))

imsave(save_path + os.sep + "regression_kym1.tiff", img_as_ubyte(app.kymograph_1_w_line))
imsave(save_path + os.sep + "regression_kym2.tiff", img_as_ubyte(app.kymograph_2_w_line))
