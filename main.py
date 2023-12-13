import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def SURF():
    root = os.getcwd()
    imgPath = os.path(root, 'demoImages/tesla.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    hessianThreshold = 3000
    surf = cv.