# lane-detecion

### ---Geometric Perspective.py---

```

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

```

##### 각종 라이브러리들을 호출합니다.

```
img = cv2.VideoCapture('line.mp4')
net, frame = img.read()
height, width, channel = frame.shape
srcPoint = np.array([[550, 500], [730, 500], [1000, 720], [200, 720]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
```

##### line.mp4 파일을 호출합니다.
##### frame 에다가 line.mp4의 이미지를 저장합니다.
##### line.mp4 영상의 세로, 가로, 색변수를 height, width, channel 변수에 저장합니다.
##### 영상에서 평면으로 변환할 부분의 좌표를 입력합니다.
##### 변환된 영상이 어떤 크기를 가질지 지정합니다.

```
while True:
    net, frame = img.read()
    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    dst = cv2.warpPerspective(frame, matrix, (width, height))
```
##### line.mp4 에서 추출한 img 변수를 frame 에 저장합니다.
##### 


```
    
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    plt.figure(figsize=(10,8))
    plt.imshow(dst, cmap='gray')
    plt.show()
```
