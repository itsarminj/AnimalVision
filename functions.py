#functions
import cv2
import math
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

def increase_brightness(img,value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    final_hsv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_hsv

def adjust_gamma(img, gamma):
    invGamma = 1 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)
def sharpening(img):
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharpening)
    return img
    #color transformation , object detection pretrained network

def binocular_vision(image,width,height,delta):
    maxangle=(math.atan(height/(width/2)))
    delta=math.radians(delta)  #for easier operation
    #print(maxangle) #48degrees
    if abs(delta - maxangle) < math.radians(10):
        roi_corners1 = np.array([[(0,0), (width/2,height), (0,height)]], dtype=np.int32) #Order of corners is also important for more than 3 corners!!!
        roi_corners2 = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    elif delta>maxangle:
        roi_corners1 = np.array([[(0,0), (0,height),(width/2,height),(width/2-height/math.tan(delta),0)]], dtype=np.int32)
        roi_corners2 = np.array([[(width/2+height/math.tan(delta),0),(width,0),(width,height),(width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(width/2-height/math.tan(delta),0), (width/2,height), (width/2+height/math.tan(delta),0)]], dtype=np.int32)
    elif delta<maxangle:
        roi_corners1 = np.array([[(0, height-(width/2)*math.tan(delta)), (0,height), (width/2,height)]], dtype=np.int32)
        roi_corners2 = np.array([[(width, height-(width/2)*math.tan(delta)), (width,height), (width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0),(0, height-(width/2)*math.tan(delta)),(width/2,height),(width, height-(width/2)*math.tan(delta)),(width,0)]], dtype=np.int32)
    channel_count = image.shape[2]  # i.e. 3 rgb
    mask1 = np.zeros(image.shape, dtype=np.uint8)
    mask2 = np.zeros(image.shape, dtype=np.uint8)
    mask3 = np.zeros(image.shape, dtype=np.uint8)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    #print(image.shape)
    ignore_mask_color = (255,) * 3  # choose channels or weights for the color transformation
    cv2.fillPoly(mask1, roi_corners1,
                            ignore_mask_color)  # fill the ROI so it doesn't get wiped out when the mask is applied
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    masked_image1 = (cv2.bitwise_and(image, mask1))
    masked_image1 = cv2.blur(masked_image1, (9, 9))

    # roi_corners = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255,) * 3  # channel_count
    cv2.fillPoly(mask2, roi_corners2, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    masked_image2 = (cv2.bitwise_and(image, mask2))
    masked_image2 = cv2.blur(masked_image2, (9, 9))

    # roi_corners = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    ignore_mask_color = (255,) * 3
    cv2.fillPoly(mask3, roi_corners3, ignore_mask_color)
    masked_image3 = sharpening(cv2.bitwise_and(mask3, image))

    dst = cv2.add(masked_image1, masked_image2)
    dst = cv2.add(dst, masked_image3)
    return dst

def horse_binocular_vision(image,width,height,delta):
    maxangle=(math.atan(height/(width/2)))
    delta=math.radians(delta)  #for easier operation
    #print(maxangle) #48degrees
    if abs(delta - maxangle) < math.radians(10):
        roi_corners1 = np.array([[(0,0), (width/2,height), (0,height)]], dtype=np.int32) #Order of corners is also important for more than 3 corners!!!
        roi_corners2 = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    elif delta>maxangle:
        roi_corners1 = np.array([[(0,0), (0,height),(width/2,height),(width/2-height/math.tan(delta),0)]], dtype=np.int32)
        roi_corners2 = np.array([[(width/2+height/math.tan(delta),0),(width,0),(width,height),(width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(width/2-height/math.tan(delta),0), (width/2,height), (width/2+height/math.tan(delta),0)]], dtype=np.int32)
    elif delta<maxangle:
        roi_corners1 = np.array([[(0, height-(width/2)*math.tan(delta)), (0,height), (width/2,height)]], dtype=np.int32)
        roi_corners2 = np.array([[(width, height-(width/2)*math.tan(delta)), (width,height), (width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0),(0, height-(width/2)*math.tan(delta)),(width/2,height),(width, height-(width/2)*math.tan(delta)),(width,0)]], dtype=np.int32)
    mask1 = np.zeros(image.shape, dtype=np.uint8)
    mask2 = np.zeros(image.shape, dtype=np.uint8)
    mask3 = np.zeros(image.shape, dtype=np.uint8)

    '''ignore_mask_color = (255,)*3 #choose channels or weights for the color transformation
    (cv2.fillPoly(mask, roi_corners1, ignore_mask_color)) # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255,)*3#channel_count
    cv2.blur(cv2.fillPoly(mask, roi_corners2, ignore_mask_color),(20,20))
    ignore_mask_color = (255,)*channel_count
    (cv2.fillPoly(mask, roi_corners3, ignore_mask_color))'''

    ignore_mask_color = (255,)*3
    roi_corners5 = np.array([[(0, height),(width/2-200, height),(width/2-200, 0),(0, 0)]], dtype=np.int32)
    roi_corners6 = np.array([[(width/2-200, height), (width/2+200, height), (width/2+200,0),(width/2-200,0)]], dtype=np.int32)
    roi_corners7 = np.array([[(width/2+200, height), (width, height), (width,0),(width/2+200,0)]], dtype=np.int32)

    cv2.fillPoly(mask1, roi_corners5, ignore_mask_color),(50,50)
    masked_image1 = cv2.bitwise_and(image, mask1)
    masked_image1 = cv2.blur(masked_image1, (20,20))

    cv2.fillPoly(mask2, roi_corners6, ignore_mask_color),(50,50)
    masked_image2 = cv2.bitwise_and(image, mask2)
    masked_image2 = sharpening(masked_image2)

    cv2.fillPoly(mask3, roi_corners7, ignore_mask_color),(50,50)
    masked_image3 = cv2.bitwise_and(image, mask3)
    masked_image3 = cv2.blur(masked_image3, (20, 20))

    dst = cv2.add(masked_image2,masked_image1)
    dst= cv2.add(masked_image3,dst)
    return dst

def put_triangle(image,width,height,delta):
    channel_count = image.shape[2]  # i.e. 3 rgb
    mask=np.zeros(image.shape, dtype=np.uint8)
    mask0 = np.zeros(image.shape, dtype=np.uint8)
    roi_corners3 = np.array([[(width/2-100, height), (width/2+100, height), (width/2,height/2-100)]], dtype=np.int32)
    roi_corners4 = np.array([[(0,0),(0,height),(width/2-100, height),(width/2,height/2-100),(width/2+100, height),(width,height),(width,0)]], dtype=np.int32)

    ignore_mask_color = (255,)*0
    cv2.fillPoly(mask, roi_corners3, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    ignore_mask_color = (255,)*3
    cv2.fillPoly(mask0, roi_corners4, ignore_mask_color)
    masked_image0 = cv2.bitwise_and(image, mask0)
    dst = cv2.add(masked_image0,masked_image)
    return dst
'''#GARBAGE
def drawtriangle(img,dim,delta):
    w=dim[0]
    y=dim[1]
    width = math.ceil(w/2)
    height=y-math.ceil(width*math.tan(delta)) #put precautions if outside window
    m=height/(-width)
    c=width  #mx+c=y
    print(list(range(y, y-height,-1)))
    print(height)
    print(y)
    for heightindex in list(range(y-10, height,-1)):
        widthindex=math.ceil((heightindex-c)/m)
        img[heightindex, 0:widthindex] = cv2.blur(img[heightindex, 0:widthindex], (3,3))
    return img

def new_drawtriangle(img,dim,delta):
    w=dim[0]
    height=dim[1]
    width = math.ceil(w/2)
    x=math.ceil(width/(math.tan(delta)))
    print(w)
    print(x)
    print(height)

    baslangic=height-x
    m=math.ceil((baslangic-height)/(0-width))
    c=x
    #print(list(range(y, y-height,-1)))
    #print(height)
    #print(c)
    #print(height)
    print(baslangic)

    for heightindex in range(baslangic, height):
        widthindex=math.ceil((heightindex-c)/m)
        img[heightindex, 0:widthindex] = cv2.blur(img[heightindex, 0:widthindex], (3,3))
    return img
'''

def FOV(image,width,height,delta):
    maxangle = (math.atan(height / (width / 2)))
    delta = math.radians(delta)  # for easier operation
    # print(maxangle) #48degrees
    if abs(delta - maxangle) < math.radians(10):
        roi_corners1 = np.array([[(0, 0), (width / 2, height), (0, height)]],
                                dtype=np.int32)  # Order of corners is also important for more than 3 corners!!!
        roi_corners2 = np.array([[(width, 0), (width / 2, height), (width, height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0, 0), (width / 2, height), (width, 0)]], dtype=np.int32)
    elif delta > maxangle:
        roi_corners1 = np.array([[(0, 0), (0, height), (width / 2, height), (width / 2 - height / math.tan(delta), 0)]],
                                dtype=np.int32)
        roi_corners2 = np.array(
            [[(width / 2 + height / math.tan(delta), 0), (width, 0), (width, height), (width / 2, height)]],
            dtype=np.int32)
        roi_corners3 = np.array([[(width / 2 - height / math.tan(delta), 0), (width / 2, height),
                                  (width / 2 + height / math.tan(delta), 0)]], dtype=np.int32)
    elif delta < maxangle:
        roi_corners1 = np.array([[(0, height - (width / 2) * math.tan(delta)), (0, height), (width / 2, height)]],
                                dtype=np.int32)
        roi_corners2 = np.array(
            [[(width, height - (width / 2) * math.tan(delta)), (width, height), (width / 2, height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0, 0), (0, height - (width / 2) * math.tan(delta)), (width / 2, height),
                                  (width, height - (width / 2) * math.tan(delta)), (width, 0)]], dtype=np.int32)
    channel_count = image.shape[2]  # i.e. 3 rgb
    mask1 = np.zeros(image.shape, dtype=np.uint8)
    mask2 = np.zeros(image.shape, dtype=np.uint8)
    mask3 = np.zeros(image.shape, dtype=np.uint8)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # print(image.shape)

    ignore_mask_color = (255,) * 3  # choose channels or weights for the color transformation
    sharpening_FOV(cv2.fillPoly(mask1, roi_corners1,
                            ignore_mask_color))  # fill the ROI so it doesn't get wiped out when the mask is applied
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    masked_image1 = (cv2.bitwise_and(image, mask1))
    masked_image1 = cv2.GaussianBlur(masked_image1, (31, 31), cv2.BORDER_DEFAULT)

    # roi_corners = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255,) * 3  # channel_count
    cv2.blur(cv2.fillPoly(mask2, roi_corners2, ignore_mask_color), (50, 50))
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    masked_image2 = (cv2.bitwise_and(image, mask2))
    masked_image2 = cv2.GaussianBlur(masked_image2, (31, 31), cv2.BORDER_DEFAULT)

    # dst=cv2.add(masked_image1,masked_image2)

    # roi_corners = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    ignore_mask_color = (255,) * 3
    sharpening_FOV(cv2.fillPoly(mask3, roi_corners3, ignore_mask_color))
    masked_image3 = (cv2.bitwise_and(mask3, image))

    dst = cv2.add(masked_image1, masked_image2)
    dst = cv2.add(dst, masked_image3)
    return dst

def sharpening_FOV(img):
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharpening)
    img = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_DEFAULT)
    return img
    #color transformation , object detection pretrained network


def slug_binocular_vision(image,width,height,delta):
    maxangle=(math.atan(height/(width/2)))
    delta=math.radians(delta)  #for easier operation
    #print(maxangle) #48degrees
    if abs(delta - maxangle) < math.radians(10):
        roi_corners1 = np.array([[(0,0), (width/2,height), (0,height)]], dtype=np.int32) #Order of corners is also important for more than 3 corners!!!
        roi_corners2 = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    elif delta>maxangle:
        roi_corners1 = np.array([[(0,0), (0,height),(width/2,height),(width/2-height/math.tan(delta),0)]], dtype=np.int32)
        roi_corners2 = np.array([[(width/2+height/math.tan(delta),0),(width,0),(width,height),(width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(width/2-height/math.tan(delta),0), (width/2,height), (width/2+height/math.tan(delta),0)]], dtype=np.int32)
    elif delta<maxangle:
        roi_corners1 = np.array([[(0, height-(width/2)*math.tan(delta)), (0,height), (width/2,height)]], dtype=np.int32)
        roi_corners2 = np.array([[(width, height-(width/2)*math.tan(delta)), (width,height), (width/2,height)]], dtype=np.int32)
        roi_corners3 = np.array([[(0,0),(0, height-(width/2)*math.tan(delta)),(width/2,height),(width, height-(width/2)*math.tan(delta)),(width,0)]], dtype=np.int32)
    channel_count = image.shape[2]  # i.e. 3 rgb
    mask = np.zeros(image.shape, dtype=np.uint8)

    ignore_mask_color = (255,)*0 #choose channels or weights for the color transformation
    sharpening(cv2.fillPoly(mask, roi_corners1, ignore_mask_color)) # fill the ROI so it doesn't get wiped out when the mask is applied
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    #masked_image1 = (cv2.bitwise_and(image, mask))

    #roi_corners = np.array([[(width,0), (width/2,height), (width,height)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    ignore_mask_color = (255,)*0#channel_count
    (cv2.fillPoly(mask, roi_corners2, ignore_mask_color),(20,20))
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    #masked_image2 = (cv2.bitwise_and(image, mask))

    #roi_corners = np.array([[(0,0), (width/2,height), (width,0)]], dtype=np.int32)
    ignore_mask_color = (255,)*3
    sharpening(cv2.fillPoly(mask, roi_corners3, ignore_mask_color))

    roi_corners4 = np.array([[(width/2-440, 0), (width/2+440, 0), (width/2,height/2+90)]], dtype=np.int32)
    ignore_mask_color = (255,)*0
    sharpening(cv2.fillPoly(mask, roi_corners4, ignore_mask_color))
    masked_image3 = (cv2.bitwise_and(image, mask))
    #dst = cv2.add(masked_image1,masked_image2,masked_image3)
    dst=masked_image3
    return dst
