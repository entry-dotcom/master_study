#　Computation of image preprocessing.

import cv2
import numpy as np
from math import pi
import time

# Gauss smooth
def GaussSmooth(img_data, ksize, sigmaX):
    print(img_data.shape)
    # Make the four-dimensional data
    if img_data.ndim==4:
        channel = img_data.shape[0]
        num_of_data = img_data.shape[1]
        #　ぼかした後の画像をまとめるnumpy配列
        gauss_img = np.empty(shape=(channel, num_of_data, 784))
        
        for i in range(channel):
            for j in range(num_of_data):
                # Spatial filtering
                dst = cv2.GaussianBlur(img_data[i][j].reshape(28,28), (ksize,ksize), sigmaX)
                # output
                # 結果を出力
                gauss_img[i][j] = dst.reshape(-1,784)
    
    else:
        num_of_data = img_data.shape[0]
        gauss_img = np.empty(shape=(num_of_data, 784))

        # Gauss smoothing phase
        for i in range(num_of_data):
            # Spatial filtering
            dst = cv2.GaussianBlur(img_data[i].reshape(28,28), (ksize,ksize), sigmaX)
            # output
            # 結果を出力
            gauss_img[i] = dst.reshape(-1,784)
        
    return gauss_img



# Image differentiation
def ImageDiff(img_data):
    # kernel definition
    kernel_x = np.array([[0, -1, 0],
                             [0, 0, 0],
                             [0, 1, 0]])

    kernel_y = np.array([[0, 0, 0],
                            [-1, 0, 1],
                            [0, 0, 0]])
    
    # Definition channnel and number of data
    num_of_data = img_data.shape[0]
    
    xdiff_img = np.empty(shape=(num_of_data,784))
    ydiff_img = np.empty(shape=(num_of_data,784))
    abs_img = np.empty(shape=(num_of_data,784))
    
    #　Image differentiation phase
    for i in range(num_of_data):
        if i%5000==0:
            print("{}まで終了".format(i))
            now = time.strftime('%Y/%m/%d %H:%M:%S')
            print(now)
        gray_x = cv2.filter2D(img_data[i].reshape(28,28), cv2.CV_64F, kernel_x)
        gray_y = cv2.filter2D(img_data[i].reshape(28,28), cv2.CV_64F, kernel_y)
        dst = gray_x ** 2 + gray_y ** 2

        # output result
        xdiff_img[i] = gray_x.reshape(-1,784)[0]
        ydiff_img[i] = gray_y.reshape(-1,784)[0]
        abs_img[i] = dst.reshape(-1,784)[0]
    
    return xdiff_img, ydiff_img, abs_img



# Computation of Fourier descriptor of edge histogram
def Fourier(xdiff_img, ydiff_img, abs_img, Q):
    
    # Definition channnel and number of data
    channel = 2*Q+1
    num_of_data = xdiff_img.shape[0]
    fourier_img = np.empty(shape=(channel, num_of_data, 28*28))
    
    # Epsilon (constant to avoid division by zero)
    ep = 10 ** -12

    # Computation of Fourier descriptor of edge histogram and output
    for i in range(Q+1):
        dq = ((abs_img+ep) ** 0.5/(2*pi)) * (((xdiff_img - ydiff_img*1j)/(abs_img+ep) ** 0.5) ** i)
        
        # Show processing time
        print("{}まで終了".format(i))
        now = time.strftime('%Y/%m/%d %H:%M:%S')
        print(now)
        print(dq.real.shape)
        if i==0:
            fourier_img[i] = dq.real
        else:
            fourier_img[2*i-1] = dq.real
            fourier_img[2*i] = dq.imag
    
    print(fourier_img.shape)
    
    return fourier_img.reshape(channel, num_of_data, 28, 28)