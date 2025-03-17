from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands untuk deteksi jari
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks keypoints YOLOv8 yang ingin ditampilkan (termasuk kaki)
KEYPOINTS_TO_SHOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# Nama bagian tubuh berdasarkan dataset COCO Keypoints
BODY_PARTS = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Knee",
    12: "Right Knee",
    13: "Left Ankle",
    14: "Right Ankle",
    15: "Left Foot",
    16: "Right Foot",
    17: "Mouth"
}

# Warna untuk setiap titik
COLORS = {
    0: (0, 0, 255),  # Hidung - Merah
    1: (255, 0, 0),  # Mata kiri - Biru
    2: (255, 0, 0),  # Mata kanan - Biru
    3: (0, 255, 255),  # Telinga kiri - Kuning
    4: (0, 255, 255),  # Telinga kanan - Kuning
    5: (0, 255, 0),  # Pundak kiri - Hijau
    6: (0, 255, 0),  # Pundak kanan - Hijau
    7: (255, 165, 0),  # Siku kiri - Oranye
    8: (255, 165, 0),  # Siku kanan - Oranye
    9: (128, 0, 128),  # Pergelangan tangan kiri - Ungu
    10: (128, 0, 128),  # Pergelangan tangan kanan - Ungu
    11: (0, 255, 255),  # Lutut kiri - Kuning
    12: (0, 255, 255),  # Lutut kanan - Kuning
    13: (255, 140, 0),  # Pergelangan kaki kiri - Coklat
    14: (255, 140, 0),  # Pergelangan kaki kanan - Coklat
    15: (0, 165, 255),  # Kaki kiri - Oranye gelap
    16: (0, 165, 255),  # Kaki kanan - Oranye gelap
    17: (255, 20, 147)  # Mulut - Pink
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan loop jika kamera tidak bisa membaca frame

    # Konversi frame ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi pose dengan YOLOv8
    results = model(frame)

    for result in results:
        # Ambil keypoints
        keypoints = result.keypoints.xy.cpu().numpy()

        # Gambar titik tubuh utama dan beri label
        for kp_idx in KEYPOINTS_TO_SHOW:
            if kp_idx < len(keypoints[0]):  # Pastikan indeks valid
                x, y = keypoints[0][kp_idx]
                color = COLORS.get(kp_idx, (255, 255, 255))  # Default putih jika tidak ada warna
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)  # Gambar titik

                # Tambahkan teks label nama bagian tubuh
                label = BODY_PARTS.get(kp_idx, "Unknown")
                cv2.putText(frame, label, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Deteksi tangan dengan MediaPipe Hands
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Jari tangan dengan warna kuning
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose + Hand Tracking with Labels", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands untuk deteksi jari
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Indeks keypoints YOLOv8 yang ingin ditampilkan (termasuk kaki)
KEYPOINTS_TO_SHOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# Nama bagian tubuh berdasarkan dataset COCO Keypoints
BODY_PARTS = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Knee",
    12: "Right Knee",
    13: "Left Ankle",
    14: "Right Ankle",
    15: "Left Foot",
    16: "Right Foot",
    17: "Mouth"
}

# Warna untuk setiap titik
COLORS = {
    0: (0, 0, 255),  # Hidung - Merah
    1: (255, 0, 0),  # Mata kiri - Biru
    2: (255, 0, 0),  # Mata kanan - Biru
    3: (0, 255, 255),  # Telinga kiri - Kuning
    4: (0, 255, 255),  # Telinga kanan - Kuning
    5: (0, 255, 0),  # Pundak kiri - Hijau
    6: (0, 255, 0),  # Pundak kanan - Hijau
    7: (255, 165, 0),  # Siku kiri - Oranye
    8: (255, 165, 0),  # Siku kanan - Oranye
    9: (128, 0, 128),  # Pergelangan tangan kiri - Ungu
    10: (128, 0, 128),  # Pergelangan tangan kanan - Ungu
    11: (0, 255, 255),  # Lutut kiri - Kuning
    12: (0, 255, 255),  # Lutut kanan - Kuning
    13: (255, 140, 0),  # Pergelangan kaki kiri - Coklat
    14: (255, 140, 0),  # Pergelangan kaki kanan - Coklat
    15: (0, 165, 255),  # Kaki kiri - Oranye gelap
    16: (0, 165, 255),  # Kaki kanan - Oranye gelap
    17: (255, 20, 147)  # Mulut - Pink
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan loop jika kamera tidak bisa membaca frame

    # Konversi frame ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi pose dengan YOLOv8
    results = model(frame)

    for result in results:
        # Ambil keypoints
        keypoints = result.keypoints.xy.cpu().numpy()

        # Gambar titik tubuh utama dan beri label
        for kp_idx in KEYPOINTS_TO_SHOW:
            if kp_idx < len(keypoints[0]):  # Pastikan indeks valid
                x, y = keypoints[0][kp_idx]
                color = COLORS.get(kp_idx, (255, 255, 255))  # Default putih jika tidak ada warna
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)  # Gambar titik

                # Tambahkan teks label nama bagian tubuh
                label = BODY_PARTS.get(kp_idx, "Unknown")
                cv2.putText(frame, label, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Deteksi tangan dengan MediaPipe Hands
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Jari tangan dengan warna kuning
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose + Hand Tracking with Labels", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
