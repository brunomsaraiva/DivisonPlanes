import cv2
import math
import numpy as np
import tkinter.filedialog as fd
from matplotlib import pyplot as plt
from skimage.draw import line
from skimage.io import imread, imsave
from skimage.filters import threshold_isodata
from skimage.color import gray2rgb
from skimage.util import img_as_float, img_as_uint
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from sklearn.decomposition import PCA
from scipy import stats, ndimage
from scipy import odr
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel
from sklearn.linear_model import LinearRegression


class AngleManager(object):

    def __init__(self):
        # Elyra settings
        self.pixel_size = 32.2
        self.z_step = 110

        self.kymograph_1 = None
        self.kymograph_2 = None
        self.filtered_kym1 = None
        self.filtered_kym2 = None
        self.kymograph_1_coords = None
        self.kymograph_2_coords = None
        self.kymograph_1_w_line = None
        self.kymograph_2_w_line = None
        self.kym1_axis = None
        self.kym2_axis = None
        self.kym1_angle = None
        self.kym2_angle = None
        self.angle_diff = None

    def load_kymographs(self, path_kym1=None, path_kym2=None):

        if path_kym1 is None:
            path_kym1 = fd.askopenfilename()
        self.kymograph_1 = imread(path_kym1)

        if path_kym2 is None:
            path_kym2 = fd.askopenfilename()
        self.kymograph_2 = imread(path_kym2)

    def find_minimum_bounding_box(self, kym):

        points = []
        nonzero_points = np.nonzero(kym)
        for i in range(len(nonzero_points[0])):
            points.append([nonzero_points[0][i], nonzero_points[1][i]])

        from scipy.ndimage.interpolation import rotate
        pi2 = np.pi/2.

        hull_object = ConvexHull(np.array(points))
        hull_vertices = hull_object.vertices
        hull_points = []
        for i in hull_vertices:
            hull_points.append(points[i])

        hull_points = np.array(hull_points)

        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]

        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])

        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        rotations = np.vstack([np.cos(angles),
                               np.cos(angles-pi2),
                               np.cos(angles+pi2),
                               np.cos(angles)]).T

        rotations = rotations.reshape((-1, 2, 2))
        rot_points = np.dot(rotations, hull_points.T)

        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

    def points_distance(self, point1, point2):
        x0 = point1[0]
        y0 = point1[1]
        x1 = point2[0]
        y1 = point2[1]

        return math.sqrt((x1-x0)**2 + (y1-y0)**2)

    def find_length_axis(self, rectangle_coords):
        #length is assumed to be the longest central axis

        left_bot = []
        left_top = []
        right_bot = []
        right_top = []

        tmp = rectangle_coords[rectangle_coords[:,0].argsort()]

        # parallel to the y axis
        if tmp[1][0] == tmp[2][0]:
            left_top = tmp[0]
            right_bot = tmp[3]

            if tmp[1][1] <= tmp[2][1]:
                left_bot = tmp[1]
                right_top= tmp[2]
            else:
                left_bot = tmp[2]
                right_top = tmp[1]

        else:
            if tmp[0][1] <= tmp[1][1]:
                left_top = tmp[0]
                right_top = tmp[1]
            else:
                left_top = tmp[1]
                right_top = tmp[0]
            if tmp[2][1] <= tmp[3][1]:
                left_bot = tmp[2]
                right_bot = tmp[3]
            else:
                left_bot = tmp[3]
                right_bot = tmp[2]

        print("LT:", left_top, "LB", left_bot, "RT", right_top, "RB", right_bot)

        if self.points_distance(left_bot, right_bot) >= self.points_distance(left_bot, left_top):
            orientation = "left2right"
            x0 = int((left_top[0]+left_bot[0])/2)
            y0 = int((left_top[1]+left_bot[1])/2)
            x1 = int((right_top[0]+right_bot[0])/2)
            y1 = int((right_top[1]+right_bot[1])/2)

        else:
            orientation = "bot2top"
            x0 = int((left_bot[0]+right_bot[0])/2)
            y0 = int((left_bot[1]+right_bot[1])/2)
            x1 = int((left_top[0]+right_top[0])/2)
            y1 = int((left_top[1]+right_top[1])/2)

        return [[x0, y0], [x1, y1], orientation]

    def lin_func(self, params, x):
        return params[0] * x + params[1]

    def linear_function(self, p, x):
        m, c = p
        return m*x + c

    def pcamethod(self, filtered_kym):
        # get x and y points of binary image
        x, y = np.nonzero(filtered_kym)
        x = [[val] for val in x]
        y = [[val] for val in y]

        coords = np.concatenate((x, y), axis=1)

        pca = PCA(n_components=1)
        pca.fit(coords)

        pos_x, pos_y = pca.mean_
        eigenvector_x, eigenvector_y = pca.components_[0]
        eigenval = pca.explained_variance_[0]

        return [[pos_x-eigenvector_x*eigenval, pos_y-eigenvector_y*eigenval], [pos_x+eigenvector_x*eigenval, pos_y+eigenvector_y*eigenval]]

    def linreg(self, filtered_kym):

        linear_model = odr.Model(self.linear_function)
        x_points, y_points = np.nonzero(filtered_kym)
        mydata = odr.RealData(x_points, y_points)
        myodr = odr.ODR(mydata, linear_model, beta0=[0., 1.])
        myodr.run()
        m, b = myodr.output.beta
        print(m, b)

        if b < 0:
            y0 = 0
            x0 = int(-b/m)
            y1 = 1
            x1 = int((1-b)/m)
        elif b == 0:
            x0 = 0
            y0 = 0
            x1 = 1
            y1 = m
        else:
            y0 = int(b) - 1
            x0 = int((y0-b)/m)
            y1 = int(b) - 2
            x1 = int((y1-b)/m)


        return [[x0, y0], [x1, y1]]


    def compute_coords(self, method="Box"):

        # kym1_sept_threshold = threshold_isodata(self.kymograph_1)
        # kym2_sept_threshold = threshold_isodata(self.kymograph_2)

        kym1_sept_threshold = np.percentile(self.kymograph_1.flatten(), 95)
        kym2_sept_threshold = np.percentile(self.kymograph_2.flatten(), 95)

        # self.filtered_kym1 = ndimage.morphology.distance_transform_edt(self.filtered_kym1) > 1.2
        # self.filtered_kym2 = ndimage.morphology.distance_transform_edt(self.filtered_kym2) > 1.2

        self.filtered_kym1 = self.kymograph_1 > kym1_sept_threshold
        self.filtered_kym2 = self.kymograph_2 > kym2_sept_threshold

        if method == "Box":

            self.kym1_rectangle_coords = self.find_minimum_bounding_box(self.filtered_kym1)
            self.kym2_rectangle_coords = self.find_minimum_bounding_box(self.filtered_kym2)

            self.kym1_axis = self.find_length_axis(self.kym1_rectangle_coords)
            self.kym2_axis = self.find_length_axis(self.kym2_rectangle_coords)

        elif method == "LinReg":
            self.kym1_axis = self.linreg(self.filtered_kym1)
            self.kym2_axis = self.linreg(self.filtered_kym2)

        elif method == "PCA":
            self.kym1_axis = self.pcamethod(self.filtered_kym1)
            self.kym2_axis = self.pcamethod(self.filtered_kym2)

    def calculate_line_points(self, kym, kym_axis):

        x0, y0 = kym_axis[0]
        x1, y1 = kym_axis[1]
        x_points, y_points = line(int(x0), int(y0), int(x1), int(y1))

        max_x, max_y = kym.shape


        fixed_x_points = []
        fixed_y_points = []

        for i in range(len(x_points)):

            if 0 < x_points[i] < max_x and 0 < y_points[i] < max_y:
                fixed_x_points.append(x_points[i])
                fixed_y_points.append(y_points[i])

        return [fixed_x_points, fixed_y_points]

    def compute_regression(self):

        kym1_line_points = self.calculate_line_points(self.kymograph_1, self.kym1_axis)
        kym2_line_points = self.calculate_line_points(self.kymograph_2, self.kym2_axis)

        self.kymograph_1_w_line = rescale_intensity(img_as_float(gray2rgb(self.kymograph_1)))
        self.kymograph_1_w_line[kym1_line_points] = (0, 1, 0)

        self.kymograph_2_w_line = rescale_intensity(img_as_float(gray2rgb(self.kymograph_2)))
        self.kymograph_2_w_line[kym2_line_points] = (0, 1, 0)

    def angle_from_points(self, kym_axis):

        x0, y0 = kym_axis[0]
        x0 = x0 * self.z_step
        y0 = y0 * self.pixel_size
        x1, y1 = kym_axis[1]
        x1 = x1 * self.z_step
        y1 = y1 * self.pixel_size

        if x0 - x1 == 0:
            angle = 0.0

        elif y0 - y1 == 0:
            angle = 90.0

        else:
            if y1 > y0:
                if x1 > x0:
                    direction = -1
                    opposite = x1 - x0
                    adjacent = y1 - y0
                else:
                    direction = 1
                    opposite = x0 - x1
                    adjacent = y1 - y0

            elif y0 > y1:
                if x1 > x0:
                    direction = 1
                    opposite = x1 - x0
                    adjacent = y0 - y1
                else:
                    direction = -1
                    opposite = x0 - x1
                    adjacent = y0 - y1

            angle = math.degrees(math.atan(opposite/adjacent)) * direction

        return angle

    def compute_angles(self):

        self.kym1_angle = self.angle_from_points(self.kym1_axis)
        self.kym2_angle = self.angle_from_points(self.kym2_axis)

        self.angle_diff = abs(self.kym1_angle - self.kym2_angle)
        if self.angle_diff > 90.0:
            self.angle_diff = 180 - self.angle_diff

