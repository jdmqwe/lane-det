import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

img = cv2.VideoCapture('line.mp4')
net, frame = img.read()
height, width, channel = frame.shape
#srcPoint = np.array([[550, 500], [730, 500], [1000, 720], [200, 720]], dtype=np.float32)
srcPoint = np.array([[550, 500], [730, 500], [1000, 720], [200, 720]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


while True:
    net, frame = img.read()
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    dst = cv2.warpPerspective(frame, matrix, (width, height))

    
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    plt.figure(figsize=(10,8))
    plt.imshow(dst, cmap='gray')
    plt.show()