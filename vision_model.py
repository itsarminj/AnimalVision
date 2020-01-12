import numpy as np
import cv2

class Vision():
    """
    Class to call to create each model of visualization
    """

    def __init__(self, filename):
        # all the constants are set here

        self.filename = filename

        # image params
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.IMG_CHANNELS = 1

    def real_time(self):
        cap = cv2.VideoCapture(0)
        while True:
            _,frame = cap.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            lower_red = np.array([150 150 50])
            upper_red = np.array([180,255,150])

            mask = cv2.inRange(hsv, lower_red,upper_red)
            res = cv2.bitwise_and(frame,frame,mask = mask)

            cv2.imshow('frame', frame)
            cv2.imshow('res', res)

            k = cv2.waitKey(5) & 0xFF
            if k ==27:
                break
            cv2.destroyAllWindows()
            cap.release()

    def load_video(self, fname = None):
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

    def load_image(self,fname = None):
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




