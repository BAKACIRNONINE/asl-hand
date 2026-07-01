import os
import csv
import cv2
import mediapipe as mp
import shutil
from tqdm import tqdm
import random
import multiprocessing
import time

# 设置采样比例，这里设置为 0.1 表示采样 10% 的图片，你可以根据需要修改
sampling_ratio = 0.05

# Initialize the hand detection module of MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.9
)
mp_drawing = mp.solutions.drawing_utils

# Dataset directory path
dataset_dir = './ASL_Alphabet_Dataset/asl_alphabet_train/'
# CSV file save path
csv_file_path = 'dataset_info.csv'
# temp1 folder path
temp1_folder = 'temp1'
# temp folder path
temp_folder = 'temp'

# 新增标志位，用于控制是否存储 temp 及 temp1 文件夹
save_temp = False
save_temp1 = False

# Check if the temp1 folder exists. If not, create it
if save_temp1 and not os.path.exists(temp1_folder):
    os.makedirs(temp1_folder)

# Check if the temp folder exists. If not, create it
if save_temp and not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

def process_sub_dir(sub_dir_path):
    # 获取当前子目录下的所有图像文件
    image_files = [f for f in os.listdir(sub_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_image_files = len(image_files)

    # 计算需要采样的有效图像数量
    target_sampled_count = int(total_image_files * sampling_ratio)
    sampled_count = 0

    # 打乱当前子目录下所有图像文件的路径
    random.shuffle(image_files)

    sub_dir_data = []
    for file in image_files:
        if sampled_count >= target_sampled_count:
            break

        image_path = os.path.join(sub_dir_path, file)
        absolute_image_path = os.path.abspath(image_path)

        # Get the class index from the folder name
        class_folder = os.path.basename(os.path.dirname(image_path))
        class_index = ord(class_folder.upper()) - ord('A')

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {absolute_image_path}")
            continue

        # Convert the BGR image to RGB
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform hand detection
        results = hands.process(image_rgb)

        # Extract hand landmark information
        hand_landmarks_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取第一个关键点的坐标
                first_landmark = hand_landmarks.landmark[0]
                first_x, first_y, first_z = first_landmark.x, first_landmark.y, first_landmark.z

                for landmark in hand_landmarks.landmark:
                    # 平移坐标
                    new_x = landmark.x - first_x
                    new_y = landmark.y - first_y
                    new_z = landmark.z - first_z
                    hand_landmarks_data.extend([new_x, new_y, new_z])
        # If two hands are not detected, fill the remaining data with 0
        while len(hand_landmarks_data) < 2 * 21 * 3:
            hand_landmarks_data.append(0)

        # Check if landmarks are detected
        if any(value != 0 for value in hand_landmarks_data):
            # Draw landmark information
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if save_temp1:
                file_name = os.path.basename(absolute_image_path)
                new_image_path = os.path.join(temp1_folder, file_name)
                cv2.imwrite(new_image_path, image)

            # Add the image file name, classification information, and hand landmark information to the data list
            sub_dir_data.append([absolute_image_path, class_index] + hand_landmarks_data)

            # 左右映射的关键点信息
            flipped_hand_landmarks_data = []
            for i in range(0, len(hand_landmarks_data), 3):
                # 对 x 坐标取反
                flipped_hand_landmarks_data.extend([-hand_landmarks_data[i], hand_landmarks_data[i+1], hand_landmarks_data[i+2]])

            # 添加左右映射后的信息
            sub_dir_data.append([absolute_image_path, class_index] + flipped_hand_landmarks_data)

            sampled_count += 1
        else:
            if save_temp:
                file_name = os.path.basename(absolute_image_path)
                new_image_path = os.path.join(temp_folder, file_name)
                shutil.copy2(absolute_image_path, new_image_path)

    return sub_dir_data

if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    all_data = []
    sub_dirs = []
    # 遍历每个子目录
    for sub_dir in os.listdir(dataset_dir):
        sub_dir_path = os.path.join(dataset_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            sub_dirs.append(sub_dir_path)

    # 使用多进程处理子目录
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_sub_dir, sub_dirs), total=len(sub_dirs), desc="Processing Subdirectories"))
        for sub_dir_data in results:
            all_data.extend(sub_dir_data)

    # Write the data to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header of the CSV file
        header = ['Image Path', 'Class Index']
        for hand_num in range(2):
            for landmark_num in range(21):
                for coord in ['x', 'y', 'z']:
                    header.append(f"Hand{hand_num + 1}_Landmark{landmark_num}_{coord}")
        writer.writerow(header)
        # Write the data
        writer.writerows(all_data)

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时长

    print(f"CSV 文件已保存到 {csv_file_path}")
    print(f"The CSV file has been saved to {csv_file_path}")
    print(f"处理总时长: {total_time:.2f} 秒")

    # Release resources
    hands.close()