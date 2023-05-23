import cv2
import numpy as np

def perspective_transform(image):
    # 원근 변환을 위한 좌표 설정
    src_points = np.array([[570, 500], [710, 500], [1000, 720], [200, 720]], dtype=np.float32)
    dst_points = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)

    # 변환 행렬 계산
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 원근 변환 적용
    transformed_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))

    return transformed_image

def lane_detection(image):
    # 이미지 전처리 (예: 흑백 변환, 가우시안 블러 적용 등)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 캐니 엣지 검출
    edges = cv2.Canny(blurred, 25, 40)

    # ROI 설정
    height, width = image.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width*0.8, height*0.8), (width*0.6, height*0.8)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # 검출된 직선 중 중심에 가장 가까운 직선을 찾음
    if lines is not None:
        center = width // 2
        closest_line = None
        closest_distance = width
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_center = (x1 + x2) // 2
            distance = abs(line_center - center)
            if distance < closest_distance:
                closest_line = line[0]
                closest_distance = distance

        # 차선 중심 계산
        if closest_line is not None:
            x1, y1, x2, y2 = closest_line
            line_center = (x1 + x2) // 2

            # 차선 중심과의 거리 계산
            distance = (center - line_center)

            # 결과 이미지에 거리 표시
            cv2.putText(image, f"Distance: {distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(distance)
    # 결과 이미지를 화면에 출력
    cv2.imshow("Lane Detection", edges)

    return edges
# 영상 불러오기
video_path = "line.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 시점 변환 수행
    transformed_frame = perspective_transform(frame)

    # 차선 검출 및 거리 계산
    result_frame = lane_detection(transformed_frame)
    
    # 결과 출력
    cv2.imshow("Lane Detection", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()