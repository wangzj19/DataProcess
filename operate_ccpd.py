# 处理CCPD数据集，生成YOLO v5s检测数据集、LPRNet数据集

import os
from pathlib import Path
import cv2
import random
import shutil
from PIL import Image

# CCPD目录
CCPD_FOLDERS = ['base-3144', 'blur-1272', 'db-1176', 'fn-1288', 'green-3112', 'rotate-2120', 'weather-1304']
DETECTION_FOLDERS = 'D:\ExperimentEnvironment\DataProcess\CCPD-sub-01\Plate-detection'
RECOGNITION_FOLDERS = 'D:\ExperimentEnvironment\DataProcess\CCPD-sub-01\Plate-recognition'
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def divide_data(train = 8, valid = 1, test = 1):
    # 返回数字意义：0-训练集；1-验证集；2-测试集；
    random.random()
    total = train + valid + test
    train = train * 1000 / total
    valid = train + valid * 1000 / total

    key = random.randint(0, 1000)
    if (key <= train):
        return 0
    elif(key <= valid):
        return 1
    else:
        return 2

def decode_plate_name(input):
    res = ''
    char_list = input.split("_");
    res += PROVINCES[int(char_list[0])]
    res += ALPHABETS[int(char_list[1])]
    for key in range(2, len(char_list)):
        res += ADS[int(char_list[key])]
    return res

def decode_ccpd(file_name):
    attribute_list = file_name.split("-")  # 第一次分割，以减号'-'做分割
    lt, rb = attribute_list[2].split("_");
    lx, ly = list(map(int, lt.split("&")))
    rx, ry = list(map(int, rb.split("&")))
    width = rx - lx
    height = ry - ly  # bounding box的宽和高
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2  # bounding box中心点
    car_number = decode_plate_name(attribute_list[4])
    return cx, cy, width, height, car_number, (lx, ly, rx, ry)

# for yolo v5s
def copy_image(origin_path, target_path, car_number, file_end):
    image_target_path = target_path + '\\' + 'images'
    my_folder = Path(image_target_path)
    if my_folder.exists() == False:
        os.makedirs(image_target_path)
    new_full_path = image_target_path + '\\' + car_number + file_end
    shutil.copy(origin_path, new_full_path)
    return new_full_path

def generate_label(label_id, cx, cy, width, height, car_number, target_path):
    label_target_path = target_path + '\\' + 'labels'
    my_folder = Path(label_target_path)
    if my_folder.exists() == False:
        os.makedirs(label_target_path)
    txt_file = label_target_path + '\\' + car_number + ".txt"
    with open(txt_file, "w") as f:
        f.write(str(label_id) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

def process_detection(file_name, path, folder_name):
    print(f'***{file_name}, {path}, {folder_name}')

    cx, cy, width, height, car_number, _ = decode_ccpd(file_name)

    full_path = path + '\\' + file_name  # 原来文件全路径

    img = cv2.imread(full_path)
    width = width / img.shape[1]
    height = height / img.shape[0]
    cx = cx / img.shape[1]
    cy = cy / img.shape[0]

    data_type = divide_data()
    target_path = DETECTION_FOLDERS
    if(data_type == 0):
        target_path += '\\' + 'train'
    elif(data_type == 1):
        target_path += '\\' + 'valid'
    else:
        target_path += '\\' + 'test'

    # 复制文件过去，并重命名
    copy_image(full_path, target_path, car_number, ".jpg")
    # 生成label文件
    generate_label(0, cx, cy, width, height, car_number, target_path)

    # 如果是测试集则再生成一份在相应目录
    if (data_type == 2):
        test_target_path = DETECTION_FOLDERS + '\\' + folder_name + '\\' + 'test'
        copy_image(full_path, test_target_path, car_number, ".jpg")
        generate_label(0, cx, cy, width, height, car_number, test_target_path)

#  for LPRNet

def cut_resize(new_full_path, bounding_box, target_size):
    img = Image.open(new_full_path)
    img = img.crop(bounding_box)
    img = img.resize(target_size)
    img.save(new_full_path)

def process_recognition(file_name, path, folder_name):
    print(f'==={file_name}, {path}, {folder_name}')

    cx, cy, width, height, car_number, box = decode_ccpd(file_name)

    full_path = path + '\\' + file_name  # 原来文件全路径

    data_type = divide_data()
    target_path = RECOGNITION_FOLDERS
    if(data_type == 0):
        target_path += '\\' + 'train'
    elif(data_type == 1):
        target_path += '\\' + 'valid'
    else:
        target_path += '\\' + 'test'

    # 复制文件过去，并重命名
    new_full_path = copy_image(full_path, target_path, car_number, ".jpg")
    cut_resize(new_full_path, box, (94, 24))

    # 如果是测试集则再生成一份在相应目录
    if (data_type == 2):
        test_target_path = RECOGNITION_FOLDERS + '\\' + folder_name + '\\' + 'test'
        new_test_full_path = copy_image(full_path, test_target_path, car_number, ".jpg")
        cut_resize(new_test_full_path, box, (94, 24))

def do_bussiness(folder_name, now_path):
    full_path = now_path + '\\' + folder_name
    # 从文件夹全路径下遍历文件去执行操作
    for root, dirs, files in os.walk(full_path):
        for file_name in files:
            process_recognition(file_name, root, folder_name)
            process_detection(file_name, root, folder_name)

if __name__ == '__main__':
    random.seed(1)
    root_path = os.getcwd() + r'\CCPD-sub-01'
    # folders = ['base-3144', 'blur-1272', 'fn-1288', 'green-3112', 'rotate-2120', 'weather-1304']
    folders = ['db-1176']
    max_over = 50
    i = 0
    for val in folders:
        do_bussiness(val, root_path)
        i += 1
        if i > max_over:
            break
