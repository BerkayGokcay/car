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
cap2 = cv2.VideoCapture(6)  # Sağ kamera

ret0, img_cam0 = cap0.read()
ret1, img_cam1 = cap1.read()

if not ret0 or not ret1:
    print("Kameralardan görüntü alınamadı.")
    cap0.release()
    cap1.release()
    exit()

# -------- Nokta okuma / seçme --------
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

cv2.destroyAllWindows()

# Transformation matrislerini hesapla
pts_src0 = np.float32(points_cam0)
pts_src1 = np.float32(points_cam1)

width, height = 200, 150
pts_dst = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

matrix0 = cv2.getPerspectiveTransform(pts_src0, pts_dst)
matrix1 = cv2.getPerspectiveTransform(pts_src1, pts_dst)

# Sağ ve sol sabit fotoğrafları oku
left_img = cv2.imread("left.jpg")
right_img = cv2.imread("right.jpg")

if left_img is None or right_img is None:
    print("left.jpg veya right.jpg bulunamadı!")
    cap0.release()
    cap1.release()
    exit()

# Sağ ve sol döndür (yön düzelt)
left_img_resized = cv2.resize(left_img, (height, width))
right_img_resized = cv2.resize(right_img, (height, width))
left_img_rotated = cv2.rotate(left_img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
right_img_rotated = cv2.rotate(right_img_resized, cv2.ROTATE_90_CLOCKWISE)

# CANVAS ve ARAÇ
canvas_width = 600
canvas_height = 600
car_width = 200
car_height = 300
car_x = (canvas_width - car_width) // 2
car_y = (canvas_height - car_height) // 2

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2,frame2 = cap2.read()

    if not ret0 or not ret1:
        print("Kamera hatası.")
        break

    # Boş canvas
    canvas = np.full((canvas_height, canvas_width, 3), (50,50,50), dtype=np.uint8)

    # ÖN ve ARKA warp
    warped0 = cv2.warpPerspective(frame0, matrix0, (width, height))
    warped1 = cv2.warpPerspective(frame1, matrix1, (width, height))
    warped1_rotated = cv2.rotate(warped1, cv2.ROTATE_180)

    # ÖN yerleştir
    resized_front = cv2.resize(warped0, (car_width, int(car_height/2)))
    y1_front = car_y - resized_front.shape[0]
    y2_front = car_y
    x1_front = car_x
    x2_front = car_x + car_width
    if y1_front >= 0:
        canvas[y1_front:y2_front, x1_front:x2_front] = resized_front

    # ARKA yerleştir
    resized_rear = cv2.resize(warped1_rotated, (car_width, int(car_height/2)))
    y1_rear = car_y + car_height
    y2_rear = y1_rear + resized_rear.shape[0]
    if y2_rear <= canvas_height:
        canvas[y1_rear:y2_rear, x1_front:x2_front] = resized_rear

    # SOL yerleştir
    x1_left = car_x - left_img_rotated.shape[1]
    x2_left = car_x
    y1_left = car_y
    y2_left = car_y + car_height
    canvas[y1_left:y2_left, x1_left:x2_left] = cv2.resize(left_img_rotated, (x2_left - x1_left, y2_left - y1_left))

    # SAĞ yerleştir
    x1_right = car_x + car_width
    x2_right = x1_right + right_img_rotated.shape[1]
    y1_right = car_y
    y2_right = car_y + car_height
    canvas[y1_right:y2_right, x1_right:x2_right] = cv2.resize(right_img_rotated, (x2_right - x1_right, y2_right - y1_right))

    # Araç kutusu
    cv2.rectangle(canvas, (car_x, car_y), (car_x + car_width, car_y + car_height), (0,255,0), 2)

    # Gösterimler
    cv2.imshow("right",frame2)
    cv2.imshow("original", frame0)
    cv2.imshow("original1", frame1)
    cv2.imshow("Bird's Eye View with Car", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
