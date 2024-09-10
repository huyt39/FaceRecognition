import cv2


import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))


# Mở camera (0 là chỉ số của camera mặc định)
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    # Kiểm tra xem có đọc thành công không
    if not ret:
        print("Không thể đọc khung hình")
        break
    faces = app.get(frame)
    # Hiển thị khung hình
    rimg = app.draw_on(frame, faces)
    cv2.imshow('Camera', rimg)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()


