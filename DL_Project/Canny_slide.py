from __future__ import print_function
import cv2 as cv

max_lowThreshold = 255
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    # dst = img_blur * (mask[:, :, None].astype(src.dtype))
    dst = img_blur * (mask.astype(src.dtype))

    cv.imshow(window_name, dst)


src = cv.imread("images/Airplane/airplane1.jpg")[:640, :1280, :]
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()
