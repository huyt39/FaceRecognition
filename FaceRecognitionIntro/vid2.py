import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def initialize_face_analysis():
    """Khởi tạo đối tượng FaceAnalysis."""
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def open_camera():
    """Mở camera và kiểm tra xem camera có mở thành công không."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        exit()
    return cap

def process_frame(frame, app):
    """Xử lý khung hình với ứng dụng FaceAnalysis và vẽ khuôn mặt trên khung hình."""
    faces = app.get(frame)
    rimg = app.draw_on(frame, faces)
    return rimg

def main():
    app = initialize_face_analysis()
    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình")
            break

        rimg = process_frame(frame, app)
        cv2.imshow('Camera', rimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
