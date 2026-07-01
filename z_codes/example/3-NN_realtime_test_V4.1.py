import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading

print("开始")
print("Start")

print("定义模型结构")
print("Define the model structure")
# 定义模型结构，需与训练时的模型结构一致
# Define the model structure, which should be consistent with the model structure during training
# Define the improved neural network model
# 定义改进后的神经网络模型
class HandLandmarkClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandLandmarkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 检查MPS是否可用
device = torch.device("cpu")
print(f"Using device: {device}")

# 加载模型
input_size = 2 * 21 * 3
num_classes = 26
model = HandLandmarkClassifier(input_size, num_classes)
# 修改这里，将 weights_only 设置为 False
model = torch.load('best_model.pth', weights_only=False)
model = model.to(device)
model.eval()

# 初始化 MediaPipe 的手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 降低分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 降低分辨率
# 尝试设置帧率为 60fps
fps = 60
cap.set(cv2.CAP_PROP_FPS, fps)

# 预测结果存储
predicted_label = None

def predict(input_tensor):
    global predicted_label
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()
        class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        predicted_label = class_names[predicted_class]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("无法读取摄像头画面")
        print("Failed to read the camera frame")
        continue
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_landmarks_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            first_landmark = hand_landmarks.landmark[0]
            first_x, first_y, first_z = first_landmark.x, first_landmark.y, first_landmark.z

            for landmark in hand_landmarks.landmark:
                new_x = landmark.x - first_x
                new_y = landmark.y - first_y
                new_z = landmark.z - first_z
                hand_landmarks_data.extend([new_x, new_y, new_z])
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        while len(hand_landmarks_data) < input_size:
            hand_landmarks_data.append(0)

        input_tensor = torch.tensor(hand_landmarks_data, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # 启动线程进行预测
        thread = threading.Thread(target=predict, args=(input_tensor,))
        thread.start()

    if predicted_label:
        cv2.putText(image, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Landmark Classification', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()