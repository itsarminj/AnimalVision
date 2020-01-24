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
        if cap.isOpened():
            while True:
                check, frame = cap.read()
                if check:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    lower_blue = np.array([35, 140, 60])
                    upper_blue = np.array([255, 255, 180])

                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    res = cv2.bitwise_and(frame, frame, mask=mask)

                    cv2.imshow('frame', frame)
                    cv2.imshow('gray', gray)
                    cv2.imshow('res', res)

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




