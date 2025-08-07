import cv2
import numpy as np
import os

def preprocess_image_for_hailo(image_path):
    img = cv2.imread(image_path)  # BGR (H, W, C)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, (320, 320))  # (H, W, C) = (480, 480, 3)
    img = img.astype(np.float32) / 255.0  # 정규화
    return img  # (480, 480, 3)

def create_calib_dataset(image_paths, save_path):
    imgs = []
    for path in image_paths:
        img = preprocess_image_for_hailo(path)
        imgs.append(img)
    calib_data = np.array(imgs)  # (N, 480, 480, 3)
    np.save(save_path, calib_data)
    print(f"Calibration dataset saved to {save_path}")

if __name__ == "__main__":
    # 예: 이미지 폴더 경로
    img_folder = "/home/mjss/Downloads/yolo_new/expanded_coco_images"
    image_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')]

    save_file = "/home/mjss/Downloads/yolo_new/calib_data.npy"
    create_calib_dataset(image_files, save_file)
