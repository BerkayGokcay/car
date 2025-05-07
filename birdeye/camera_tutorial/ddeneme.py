import cv2

def list_available_cameras(max_cams=10):
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cams.append(i)
            cap.release()
    return available_cams

# Tüm açık kameraları tespit et
cameras = list_available_cameras()

if not cameras:
    print("Hiçbir kamera bulunamadı!")
    exit()

# Kamera akışlarını başlat
caps = [cv2.VideoCapture(cam_id, cv2.CAP_V4L2) for cam_id in cameras]

print(f"{len(cameras)} kamera bulundu: {cameras}")
print("Canlı akış başlatılıyor... 'q' tuşu ile çıkabilirsiniz.")

while True:
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, (320, 240))  # Görüntüleri küçült
            cv2.imshow(f"Camera {cameras[idx]}", resized)
        else:
            print(f"Kamera {cameras[idx]} görüntü veremiyor.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
