import cv2
import numpy as np
import os

points_cam0 = []
points_cam1 = []

def save_points(filename, points):
    np.savetxt(filename, np.array(points), fmt="%.2f")

def load_points(filename):
    return np.loadtxt(filename).reshape(-1, 2).tolist()

def mouse_callback_cam0(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_cam0) < 4:
        points_cam0.append([x, y])
        cv2.circle(img_cam0_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 points for Camera 0", img_cam0_copy)

def mouse_callback_cam1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_cam1) < 4:
        points_cam1.append([x, y])
        cv2.circle(img_cam1_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 points for Camera 1", img_cam1_copy)

cap0 = cv2.VideoCapture(2)  # Ön kamera
cap1 = cv2.VideoCapture(4)  # Arka kamera

ret0, img_cam0 = cap0.read()
ret1, img_cam1 = cap1.read()

if not ret0 or not ret1:
    print("Kameralardan görüntü alınamadı.")
    cap0.release()
    cap1.release()
    exit()

# --------------- NOKTA OKUMA / SEÇME ----------------

if os.path.exists("points_cam0.txt"):
    print("points_cam0.txt bulundu, dosyadan yükleniyor.")
    points_cam0 = load_points("points_cam0.txt")
else:
    img_cam0_copy = img_cam0.copy()
    cv2.imshow("Select 4 points for Camera 0", img_cam0_copy)
    cv2.setMouseCallback("Select 4 points for Camera 0", mouse_callback_cam0)
    print("ÖN KAMERA için 4 nokta seçin (saat yönünde).")
    while len(points_cam0) < 4:
        cv2.waitKey(1)
    save_points("points_cam0.txt", points_cam0)
    print("points_cam0.txt dosyasına kaydedildi.")

if os.path.exists("points_cam1.txt"):
    print("points_cam1.txt bulundu, dosyadan yükleniyor.")
    points_cam1 = load_points("points_cam1.txt")
else:
    img_cam1_copy = img_cam1.copy()
    cv2.imshow("Select 4 points for Camera 1", img_cam1_copy)
    cv2.setMouseCallback("Select 4 points for Camera 1", mouse_callback_cam1)
    print("ARKA KAMERA için 4 nokta seçin (saat yönünde).")
    while len(points_cam1) < 4:
        cv2.waitKey(1)
    save_points("points_cam1.txt", points_cam1)
    print("points_cam1.txt dosyasına kaydedildi.")

cv2.destroyAllWindows()

pts_src0 = np.float32(points_cam0)
pts_src1 = np.float32(points_cam1)

width, height = 200, 150  # bird’s eye view çıktı boyutu
pts_dst = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

matrix0 = cv2.getPerspectiveTransform(pts_src0, pts_dst)
matrix1 = cv2.getPerspectiveTransform(pts_src1, pts_dst)

print("Dönüşüm matrisleri hesaplandı. Canlı video başlıyor...")

canvas_width = 600
canvas_height = 800
canvas_center_x = canvas_width // 2
canvas_center_y = canvas_height // 2

car_width = 200
car_height = 300
car_x = canvas_center_x - car_width // 2
car_y = canvas_center_y - car_height // 2

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Kamera hatası.")
        break

    warped0 = cv2.warpPerspective(frame0, matrix0, (width, height))
    warped1 = cv2.warpPerspective(frame1, matrix1, (width, height))
    warped1_rotated = cv2.rotate(warped1, cv2.ROTATE_180)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    cv2.rectangle(canvas, (car_x, car_y), (car_x + car_width, car_y + car_height), (0, 255, 0), 3)

    resized_front = cv2.resize(warped0, (car_width, int(car_height/2)))
    y1_front = car_y - resized_front.shape[0]
    y2_front = car_y
    x1_front = car_x
    x2_front = car_x + car_width
    if y1_front >= 0:
        canvas[y1_front:y2_front, x1_front:x2_front] = resized_front

    resized_rear = cv2.resize(warped1_rotated, (car_width, int(car_height/2)))
    y1_rear = car_y + car_height
    y2_rear = y1_rear + resized_rear.shape[0]
    x1_rear = car_x
    x2_rear = car_x + car_width
    if y2_rear <= canvas_height:
        canvas[y1_rear:y2_rear, x1_rear:x2_rear] = resized_rear

    cv2.imshow("Bird's Eye View with Car", canvas)
    cv2.imshow("Camera 0", frame0)
    cv2.imshow("Camera 1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
