import numpy as np
import cv2

class Vision():
    """
    Class to call to create each model of visualization
    """

    def __init__(self, filename):
        # all the constants are set here

        self.filename = filename
        self.frame = None

        # image params
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 1
        self.kernel_size = (1, 1)
        self.resize_percent = 100 # percent of original size

    def real_time(self, *args):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while True:
                check, frame = cap.read()
                self.check = check
                self.frame = frame

                width = int(frame.shape[1] * self.resize_percent / 100)
                height = int(frame.shape[0] * self.resize_percent / 100)
                dim = (width, height)
                # resize image
                resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                frames = []
                res =[]

                for i in args:
                    temp = np.array(i(frame=resized))
                    try:
                        temp = self.gaussian_blur(frame=temp)
                    except:
                        pass
                        print("Kernel didn't work")
                    frames.append(temp)

                frames = np.concatenate([frames[:]], axis=1)

                cv2.imshow('Vision', np.concatenate(frames, axis=1))

                if check:
                    key = cv2.waitKey(50)
                    if key == ord('q'):
                        break
                else:
                    print('Frame not available')
                    print(cap.isOpened())

        cv2.destroyAllWindows()
        cap.release()

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

    def snake_vision(self, check = None, frame=None):
        snake_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.snakefilter = snake_filter
        return snake_filter

    def dog_vision(self, check = None, frame=None):
        dog_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.dogfilter = dog_filter
        return dog_filter

    def human(self, check = None, frame=None):
        human = frame
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





