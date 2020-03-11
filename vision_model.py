import numpy as np
import cv2
from functions import increase_brightness, binocular_vision,adjust_gamma,horse_binocular_vision, FOV #,new_drawtriangle,drawtriangle
import math
import scipy.misc
import time
from PIL import Image
import sys
import PIL
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib as mpl
import matplotlib.cm as mtpltcm


class Vision():
    """
    Class to call to create each model of visualization
    """

    def __init__(self, lmain):
        # all the constants are set here

        self.frame = None
        self.lmain = lmain


        # image params
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 1
        self.kernel_size = (1, 1)
        self.resize_percent = 100 # percent of original size
        self.dim = 0
        self.ix = 3
        self.cbmats = (
    (0.202001295331, 0.991720719265, -0.193722014597, 0,
    0.163800203026, 0.792663865514, 0.0435359314602, 0,
    0.00913336570448, -0.0132684300993, 1.00413506439, 0),

    (0.430749076295, 0.717402505462, -0.148151581757, 0,
    0.336582831043, 0.574447762213, 0.0889694067435, 0,
    -0.0236572929497, 0.0275635332006, 0.996093759749, 0),

    (0.971710712275, 0.112392320487, -0.0841030327623, 0,
    0.0219508442818, 0.817739672383, 0.160309483335, 0,
    -0.0628595877201, 0.880724870686, 0.182134717034, 0),

    (0.316086131719, 0.854894118151, -0.170980249869, 0,
    0.250572926562, 0.683189199376, 0.0662378740621, 0,
    -0.00735450321111, 0.00718184676374, 1.00017265645, 0),

    (0.299, 0.587, 0.114, 0,
    0.299, 0.587, 0.114, 0,
    0.299, 0.587, 0.114, 0),

    (0.340841450085, 0.580912815482, 0.0782457344332, 0,
    0.340841450085, 0.580912815482, 0.0782457344332, 0,
    0.340841450085, 0.580912815482, 0.0782457344332, 0),

    (0.150317227739, 0.722407271325, 0.127275500935, 0,
    0.150317227739, 0.722407271325, 0.127275500935, 0,
    0.150317227739, 0.722407271325, 0.127275500935, 0),

    (0.0336717653952, 0.114595364984, 0.851732869621, 0,
    0.0336717653952, 0.114595364984, 0.851732869621, 0,
    0.0336717653952, 0.114595364984, 0.851732869621, 0)
    )
        # CREATE THE BARS
        # FOR ANGLE BAR
        self.alpha = 70
        self.alpha_slider_max = 90
        self.title_window = 'Delta Linear Blend'
        cv2.namedWindow(self.title_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title_window, 600, 200)
        self.trackbar_name = 'Delta x %d' % self.alpha_slider_max

        def on_trackbar(val):
            alpha = val / self.alpha_slider_max
            beta = (1.0 - alpha)
            # dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
            cv2.imshow(self.title_window, 3)

        cv2.createTrackbar(self.trackbar_name, self.title_window, 70, self.alpha_slider_max, on_trackbar)
        # Show some stuff
        on_trackbar(0)

        # FOR DEPTH BAR
        self.depth = 0  # meters
        self.alpha_slider_max2 = 10  # m  complete darkness
        self.title_window2 = 'Depth linear blend'
        cv2.namedWindow(self.title_window)
        self.trackbar_name2 = 'Depth x %d' % self.alpha_slider_max2

        def on_trackbar2(val):
            alpha2 = val / self.alpha_slider_max2
            beta2 = (1.0 - self.depth)
            # dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
            cv2.imshow(self.title_window, 3)

        cv2.createTrackbar(self.trackbar_name2, self.title_window, 0, self.alpha_slider_max2, on_trackbar2)
        # Show some stuff
        on_trackbar2(0)

    def on_trackbar(self,val):
        alpha = val / self.alpha_slider_max
        beta = ( 1.0 - alpha )
        #dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
        cv2.imshow(self.title_window, 3)

    def real_time(self, *args):
        camera_index = 0
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        i = 0
        if cap.isOpened():
            check, frame = cap.read()
            self.check = check
            self.frame = frame

            width = int(frame.shape[1] * self.resize_percent / 100)
            height = int(frame.shape[0] * self.resize_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            #FOV effect
            split1 = resized[:, np.r_[0:120]]
            split2 = resized[:, np.r_[120:520]]
            split3 = resized[:, np.r_[520:640]]

            split1 = np.repeat(split1, 2, axis=1)
            split1 = np.hsplit(split1, 2)[1]
            split3 = np.repeat(split3, 2, axis=1)
            split3 = np.hsplit(split3, 2)[0]

            # split1 = cv2.GaussianBlur(split1, (9,9), cv2.BORDER_DEFAULT)
            # split3 = cv2.GaussianBlur(split3, (9, 9), cv2.BORDER_DEFAULT)

            img_final = np.concatenate([split1, split2, split3], axis=1)
            img = Image.fromarray(np.array(args[i](frame=img_final)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            self.lmain.after(50, self.real_time(*args))


    def load_video(self, fname=None):
        if fname is None and self.filename is None:
            assert False, 'Please specify a valid path to get the model from.'
        if fname is None:
            fname = self.filename

        cap = cv2.VideoCapture(fname + '.mov')

        while (cap.isOpened()):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def load_image(self, fname=None):
        if fname is None and self.filename is None:
            assert False, 'Please specify a valid path to get the model from.'
        if fname is None:
            fname = self.filename

        img = cv2.imread(fname + '.jpg', 1)

        # Percentage Scaling - should probably change this to predetermined scale
        # resize image
        scale_percent = 30  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        self.img = img

        return img

    def show_image(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Snake(self, check = None, frame=None):

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # initialize the colormap (jet)
        colormap = mpl.cm.jet_r
        # add a normalization
        cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
        # init the mapping
        scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)
        # ...
        # in the main display loop:
        snake_filter = scalarMap.to_rgba(frame)
        snake_filter = (snake_filter)*255
        self.snakefilter = snake_filter
        return snake_filter

    def Dog(self, check = None, frame=None):
        depth = cv2.getTrackbarPos(self.trackbar_name2, self.title_window)  # for color transformation
        try:
            gamma = 1 / (depth + 1)  # for changing light
        except:
            gamma = 0
        width = int(frame.shape[1] * self.resize_percent / 100)
        height = int(frame.shape[0] * self.resize_percent / 100)
        dim = (width, height)
        frame = increase_brightness(frame, 20)  # fish have bigger eyes, they collect more light
        frame = adjust_gamma(frame, gamma)  # changes gamma(light) depending on depth
        dogvision = frame
        frame_rgb = dogvision[:, :, ::-1]
        img = scipy.misc.toimage(frame_rgb)
        outfile = 'frame.jpg'
        cbmat = self.cbmats[self.ix]
        imgx = img.convert('RGB', cbmat)
        imgx.save(outfile)
        dogvision = np.asarray(PIL.Image.open(outfile))
        dogvision = dogvision[:, :, ::-1]

        # FOV effect
        delta = cv2.getTrackbarPos(self.trackbar_name, self.title_window)  # for changing the angle
        frame = FOV(frame, width, height, delta)  # changes the field of view
        self.dogvision = frame
        return frame

    def Human(self, check = None, frame=None):
        human = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.human = human
        return human

    def gaussian_blur(self, frame=None):
        blur = cv2.blur(frame, self.kernel_size)
        self.blur = blur
        return blur

    def fisheye(self, frame = None):
        K = np.array([[689.21, 0., 1295.56],
                      [0., 690.48, 942.17],
                      [0., 0., 1.]])
        D = np.array([0., 0., 0., 0.])

        # use Knew to scale the output
        Knew = K.copy()
        Knew[(0, 1), (0, 1)] = 0.4 * Knew[(0, 1), (0, 1)]

        img_undistorted = cv2.fisheye.undistortImage(frame, K, D=D, Knew=Knew)
        self.fisheye = img_undistorted
        return img_undistorted

    def concat_tile_resize(self, im_list_2d, interpolation=cv2.INTER_CUBIC):
        im_list_v = [self.hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
        return self.vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    @staticmethod
    def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        w_min = min(im.shape[1] for im in im_list)
        im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                          for im in im_list]
        return cv2.vconcat(im_list_resize)

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def sharpening(self, check=None, frame=None):
        kernel_sharpening = np.array([[-1,-1,-1], [-1, 11,-1],[-1,-1,-1]])
        sharpening_filter = cv2.filter2D(frame, -1, kernel_sharpening)
        self.sharpeningfilter = sharpening_filter
        return sharpening_filter
        #color transformation , object detection pretrained networ

    def Fish(self, check=None, frame=None):
        delta = cv2.getTrackbarPos(self.trackbar_name, self.title_window) # for changing the angle
        depth = cv2.getTrackbarPos(self.trackbar_name2, self.title_window) #for color transformation
        gamma=1/(depth+1) #for changing light
        width = int(frame.shape[1] * self.resize_percent / 100)
        height = int(frame.shape[0] * self.resize_percent / 100)
        dim = (width, height)
        width = dim[0] #width of the window
        height = dim[1] #height of the window
        frame = binocular_vision(frame,width,height,delta) #changes the field of view
        frame = increase_brightness(frame,20) #fish have bigger eyes, they collect more light
        frame = adjust_gamma(frame, gamma) #changes gamma(light) depending on depth
        fishvision = frame
        self.fishvision = fishvision
        return fishvision

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def Horse(self, check=None, frame=None):
        delta = 50 # for changing the angle
        depth = cv2.getTrackbarPos(self.trackbar_name2, self.title_window) #for color transformation
        try:
            gamma=1/(depth+1) #for changing light
        except:
            gamma = 0
        width = int(frame.shape[1] * self.resize_percent / 100)
        height = int(frame.shape[0] * self.resize_percent / 100)
        dim = (width, height)
        width=dim[0] #width of the window
        height=dim[1] #height of the window
        frame = horse_binocular_vision(frame,width,height,delta) #changes the field of view
        frame = increase_brightness(frame,20) #fish have bigger eyes, they collect more light
        frame = adjust_gamma(frame, gamma) #changes gamma(light) depending on depth
        horsevision = frame
        frame_rgb = horsevision[:, :, ::-1]
        img = scipy.misc.toimage(frame_rgb)
        outfile='frame.jpg'
        #img = Image.open(r"/Users/umutfidan/Desktop/AnimalVision-master4/frame.jpg")
        cbmat = self.cbmats[self.ix]
        imgx = img.convert('RGB',cbmat)
        #dog_filter = np.asarray(imgx, dtype="int32" )
        imgx.save(outfile)
        #exit()
        horsevision = np.asarray(PIL.Image.open(outfile))
        horsevision = horsevision[:, :, ::-1]
        self.horsevision = horsevision
        return horsevision

    def Fly(self, frame=None, size=(9, 8, 7, 6, 5, 6, 7, 8, 9)):
        im1 = frame
        tile_list = []

        for i in size:
            tile_list.append([im1]*i)

        im_tile = self.concat_tile_resize(tile_list)

        width = int(frame.shape[1])
        height = int(frame.shape[0])
        dim = (width, height)
        # resize image to fir concatenation needs
        im_tile_resize = cv2.resize(im_tile, dim, interpolation=cv2.INTER_AREA)

        tile_size = np.shape(im_tile_resize)
        overlay = np.zeros(tile_size, np.uint8)

        # Ellipse Size
        axesLength = (100, 50)
        center_coordinates = (int(tile_size[1]/2), int(tile_size[0]/2))
        axesLength = (int(tile_size[1]/2), int(tile_size[0]/2))
        angle = 0
        startAngle = 0
        endAngle = 360
        color = (255, 255, 255)  # BGR
        thickness = -1

        cv2.ellipse(overlay, center_coordinates, axesLength,
                            angle, startAngle, endAngle, color, thickness, 8)


        output = cv2.bitwise_and(im_tile_resize, overlay)

        self.fly = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)


