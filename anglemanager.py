import math
import numpy as np
import tkinter.filedialog as fd
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_isodata
from skimage.color import gray2rgb
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from scipy import stats, ndimage
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel
from sklearn.linear_model import LinearRegression


class AngleManager(object):

    def __init__(self):

        self.kymograph_1 = None
        self.kymograph_2 = None
        self.filtered_kym1 = None
        self.filtered_kym2 = None
        self.kymograph_1_coords = None
        self.kymograph_2_coords = None
        self.kymograph_1_w_line = None
        self.kymograph_2_w_line = None

    def load_kymographs(self):

        self.kymograph_1 = imread(fd.askopenfilename())
        self.kymograph_2 = imread(fd.askopenfilename())

    def compute_coords(self, percentage):

        kym1_values = sorted(self.kymograph_1.flatten(), reverse=True)
        kym2_values = sorted(self.kymograph_2.flatten(), reverse=True)

        kym1_y_max = len(kym1_values)
        kym2_y_max = len(kym2_values)

        kym1_sept_threshold = kym1_values[int(kym1_y_max * percentage)]
        kym2_sept_threshold = kym2_values[int(kym2_y_max * percentage)]

        self.filtered_kym1 = self.kymograph_1 > kym1_sept_threshold
        self.filtered_kym2 = self.kymograph_2 > kym2_sept_threshold

        self.kymograph_1_coords = stats.linregress(np.nonzero(self.filtered_kym1))
        self.kymograph_2_coords = stats.linregress(np.nonzero(self.filtered_kym2))

    def rotation_matrices(self, step):
        """ returns a list of rotation matrixes over 180 deg
        matrixes are transposed to use with 2 column point arrays (x,y),
        multiplying after the array
        """

        result = []
        ang = 0

        while ang < 180:
            sa = np.sin(ang / 180.0 * np.pi)
            ca = np.cos(ang / 180.0 * np.pi)
            # note .T, for column points
            result.append(np.matrix([[ca, -sa], [sa, ca]]).T)
            ang = ang + step

        return result

    def bounded_value(self, minval, maxval, currval):
        """ returns the value or the extremes if outside
        """

        if currval < minval:
            return minval

        elif currval > maxval:
            return maxval

        else:
            return currval


    def bounded_point(self, x0, x1, y0, y1, p):
        tx, ty = p
        tx = self.bounded_value(x0, x1, tx)
        ty = self.bounded_value(y0, y1, ty)
        return tx, ty

    def bound_rectangle(self, points):
        """ returns a tuple (x0,y0,x1,y1,width) of the bounding rectangle
        points must be a N,2 array of x,y coords
        """

        x0, y0 = np.amin(points, axis=0)
        x1, y1 = np.amax(points, axis=0)
        a = np.min([(x1 - x0), (y1 - y0)])

        return x0, y0, x1, y1, a

    def get_kym_box(self, kym, rotations):

        nonzero_points = np.nonzero(kym)
        points = []

        for i in range(len(nonzero_points[0])):
            points.append([nonzero_points[0][i], nonzero_points[1][i]])

        width = len(points) + 1

        for rix in range(len(rotations) // 2 + 1):
            r = rotations[rix]
            nx0, ny0, nx1, ny1, nwidth = self.bound_rectangle(
                np.asarray(np.dot(points, r)))

            if nwidth < width:
                width = nwidth
                x0 = nx0
                x1 = nx1
                y0 = ny0
                y1 = ny1
                angle = rix

        print(rotations[angle])

        return (nx0, ny0, nx1, ny1), rotations[angle]

    def axes_from_rotation(self, box, rotation, container_box):
        """ sets the cell axes from the box and the rotation
        """

        x0, y0, x1, y1 = box

        # midpoints
        mx = (x1 + x0) / 2
        my = (y1 + y0) / 2

        # assumes long is X. This duplicates rotations but simplifies
        # using different algorithms such as brightness
        x_axis = [[x0, my], [x1, my]]
        y_axis = [[mx, y0], [mx, y1]]
        y_axis = np.asarray(np.dot(y_axis, rotation.T), dtype=np.int32)
        x_axis = np.asarray(np.dot(x_axis, rotation.T), dtype=np.int32)

        # check if axis fall outside area due to rounding errors
        bx0, by0, bx1, by1 = container_box
        y_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, y_axis[0])
        y_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, y_axis[1])
        x_axis[0] = \
            self.bounded_point(bx0, bx1, by0, by1, x_axis[0])
        x_axis[1] = \
            self.bounded_point(bx0, bx1, by0, by1, x_axis[1])

        x_measure = np.linalg.norm(x_axis[1] - x_axis[0])
        y_measure = np.linalg.norm(y_axis[1] - y_axis[0])

        if x_measure >= y_measure:
            return x_axis

        else:
            return y_axis

    def compute_container(self, kym):

        points = np.nonzero(kym)
        x_points = sorted(points[0])
        y_points = sorted(points[1])

        return x_points[0], y_points[0], x_points[-1], y_points[-1]

    def compute_centered_coords(self):

        kym1_sept_threshold = threshold_isodata(self.kymograph_1)
        kym2_sept_threshold = threshold_isodata(self.kymograph_2)

        self.filtered_kym1 = self.kymograph_1 > kym1_sept_threshold
        self.filtered_kym2 = self.kymograph_2 > kym2_sept_threshold

        self.filtered_kym1 = ndimage.morphology.distance_transform_edt(self.filtered_kym1) > 1
        self.filtered_kym2 = ndimage.morphology.distance_transform_edt(self.filtered_kym2) > 1

        rotations = self.rotation_matrices(2)

        kym1_box, kym1_angle = self.get_kym_box(self.filtered_kym1, rotations)
        kym2_box, kym2_angle = self.get_kym_box(self.filtered_kym2, rotations)

        kym1_container = self.compute_container(self.filtered_kym1)
        kym2_container = self.compute_container(self.filtered_kym2)

        kym1_axis = self.axes_from_rotation(kym1_box, kym1_angle, kym1_container)
        kym2_axis = self.axes_from_rotation(kym2_box, kym2_angle, kym2_container)

        self.kymograph_1_coords = kym1_axis
        self.kymograph_2_coords = kym2_axis

        # LinRegModel from sklearn

        """kym1_model = LinearRegression()
        data_points_kym1 = np.nonzero(self.filtered_kym1)
        weights_kym1 = []
        for i in range(len(data_points_kym1[0])):
            x = data_points_kym1[0][i]
            y = data_points_kym1[1][i]
            weights_kym1.append(self.filtered_kym1[x][y])
        kym1_model.fit(np.array(data_points_kym1[0]).reshape(-1, 1), np.array(data_points_kym1[1]).reshape(-1, 1).astype(float), np.array(weights_kym1))

        kym1_x0_predict = kym1_model.predict([[0]])
        kym1_x1_predict = kym1_model.predict([[self.filtered_kym1.shape[0]]])

        kym1_slope = (kym1_x1_predict - kym1_x0_predict) / float(self.filtered_kym1.shape[0])
        kym1_intercept = kym1_x1_predict

        self.kymograph_1_coords = kym1_slope, kym1_intercept

        kym2_model = LinearRegression()
        data_points_kym2 = np.nonzero(self.filtered_kym2)
        weights_kym2 = []
        for i in range(len(data_points_kym2[0])):
            x = data_points_kym2[0][i]
            y = data_points_kym2[1][i]
            weights_kym2.append(self.filtered_kym2[x][y])
        kym2_model.fit(np.array(data_points_kym2[0]).reshape(-1, 1), np.array(data_points_kym2[1]).reshape(-1, 1).astype(float), np.array(weights_kym2))

        kym2_x0_predict = kym2_model.predict([[0]])
        kym2_x1_predict = kym2_model.predict([[self.filtered_kym2.shape[0]]])

        kym2_slope = (kym2_x1_predict - kym2_x0_predict) / float(self.filtered_kym2.shape[0])
        kym2_intercept = kym2_x1_predict

        self.kymograph_2_coords = kym2_slope, kym2_intercept"""

        # ellipse fitting

        """data_points_kym1 = np.nonzero(self.filtered_kym1)
        model_kym1 = EllipseModel()
        model_kym1.estimate(np.column_stack((data_points_kym1[0], data_points_kym1[1])))

        kym1_params = model_kym1.params

        data_points_kym2 = np.nonzero(self.filtered_kym2)
        model_kym2 = EllipseModel()
        model_kym2.estimate(np.column_stack((data_points_kym2[0], data_points_kym2[1])))

        kym2_params = model_kym2.params

        kym1_theta = kym1_params[-1] # in radians
        kym1_slope = math.sin(kym1_theta) / math.cos(kym1_theta)
        kym1_intercept = kym1_params[0] - kym1_slope * kym1_params[1]

        kym2_theta = kym2_params[-1] # in radians
        kym2_slope = math.sin(kym2_theta) / math.cos(kym2_theta)
        kym2_intercept = kym2_params[0] - kym2_slope * kym2_params[1]

        self.kymograph_1_coords = [kym1_slope, kym1_intercept]
        self.kymograph_2_coords = [kym2_slope, kym2_intercept]"""

        # try openCV fitLine

        """kym1_nonzero = np.nonzero(self.filtered_kym1)
        kym1_nonzero_x = sorted(kym1_nonzero[0])
        kym1_nonzero_y = sorted(kym1_nonzero[1])

        kym2_nonzero = np.nonzero(self.filtered_kym2)
        kym2_nonzero_x = sorted(kym2_nonzero[0])
        kym2_nonzero_y = sorted(kym2_nonzero[1])

        kym1_box = kym1_nonzero_x[0], kym1_nonzero_x[-1], kym1_nonzero_y[0], kym1_nonzero_y[-1]
        kym2_box = kym2_nonzero_x[0], kym2_nonzero_x[-1], kym2_nonzero_y[0], kym2_nonzero_y[-1]

        if kym1_box[1] - kym1_box[0] >= kym1_box[3] - kym1_box[2]:

            kym = self.filtered_kym1
            new_kym = np.zeros(self.filtered_kym1.shape)

            for x_coord in range(kym.shape[0]):
                nonzero = np.nonzero(kym[x_coord])
                if len(nonzero[0]) > 0:
                    new_kym[x_coord, int(np.average(nonzero[0]))] = 1

            self.centered_kym1 = new_kym

        else:

            kym = self.filtered_kym1
            new_kym = np.zeros(self.filtered_kym1.shape)

            for y_coord in range(kym.shape[1]):
                nonzero = np.nonzero(kym[:,y_coord])
                if len(nonzero[0]) > 0:
                    new_kym[int(np.average(nonzero[0])), y_coord] = 1

            self.centered_kym1 = new_kym

        if kym2_box[1] - kym2_box[0] >= kym2_box[3] - kym2_box[2]:

            kym = self.filtered_kym2
            new_kym = np.zeros(self.filtered_kym2.shape)

            for x_coord in range(kym.shape[0]):
                nonzero = np.nonzero(kym[x_coord])
                if len(nonzero[0]) > 0:
                    new_kym[x_coord, int(np.average(nonzero[0]))] = 1

            self.centered_kym2 = new_kym

        else:

            kym = self.filtered_kym2
            new_kym = np.zeros(self.filtered_kym2.shape)

            for y_coord in range(kym.shape[1]):
                nonzero = np.nonzero(kym[:, y_coord])
                print(nonzero)
                if len(nonzero[0]) > 0:
                    new_kym[int(np.average(nonzero[0])), y_coord] = 1


            self.centered_kym2 = new_kym

        self.kymograph_1_coords = stats.linregress(np.nonzero(self.centered_kym1))
        self.kymograph_2_coords = stats.linregress(np.nonzero(self.centered_kym2))"""

        # try numpy.polyfit with weighted values


    def calculate_line_points(self, kymograph, slope, intercept):

        border = 1
        x_max = kymograph.shape[0] - border
        y_max = kymograph.shape[1] - 1

        points_x = []
        points_y = []

        for x in range(border, x_max):

            y = int(slope*x + intercept)

            if y_max > y and y > 0:
                points_x.append(x)
                points_y.append(y)

        return [points_x, points_y]

    def compute_regression(self):

        kym1_line_points = self.calculate_line_points(self.kymograph_1, self.kymograph_1_coords[0], self.kymograph_1_coords[1])
        kym2_line_points = self.calculate_line_points(self.kymograph_2, self.kymograph_2_coords[0], self.kymograph_2_coords[1])

        self.kymograph_1_w_line = rescale_intensity(img_as_float(gray2rgb(self.kymograph_1)))
        self.kymograph_1_w_line[kym1_line_points] = (0, 1, 0)

        self.kymograph_2_w_line = rescale_intensity(img_as_float(gray2rgb(self.kymograph_2)))
        self.kymograph_2_w_line[kym2_line_points] = (0, 1, 0)

    def compute_angle(self):

        pass

    def save_data(self):

        pass
