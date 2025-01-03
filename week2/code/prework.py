import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import shutil
import random

def data_Augmentation():
    # 读取去重后的数据
    deduplicated_data = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\deduplicated_data.csv')

    # 创建目标文件夹
    output_dir = r'D:\document\MCM\week2\第二周小测图片文件\raw_data'
    os.makedirs(output_dir, exist_ok=True)

    # 定义旋转角度
    angles = [0, 90, 180, 270]

    # 遍历去重后的数据，查找FileType为image/*的行
    with tqdm(total=len(deduplicated_data), desc="Processing") as pbar:
        for index, row in deduplicated_data.iterrows():
            pbar.update(1)
            if row['FileType'].startswith('image/'):
                file_name = row['FileName']
                image_path = os.path.join(r'D:\document\MCM\week2\第二周小测图片文件\2021MCM_ProblemC_Files', file_name)

            # 读取图片
            try:
                image = Image.open(image_path)

                # 进行旋转并保存
                for angle in angles:
                    rotated_image = image.rotate(angle)
                    # 生成保存的文件名
                    base_name = os.path.basename(file_name)
                    file_name_without_ext, ext = os.path.splitext(base_name)
                    rotated_image.save(os.path.join(output_dir, f"{file_name_without_ext}_rotated_{angle}{ext}"))
            except Exception as e:
                print(f"无法处理文件 {image_path}: {e}")

def split_train_test():
    # 读取去重后的数据
    deduplicated_data = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\deduplicated_data.csv')

    # 定义源图片目录和目标目录
    source_dir = r'D:\document\MCM\week2\第二周小测图片文件\raw_data'
    train_data_path = r'D:\document\MCM\week2\第二周小测图片文件\train_data'
    val_data_path = r'D:\document\MCM\week2\第二周小测图片文件\val_data'
    test_data_path = r'D:\document\MCM\week2\第二周小测图片文件\test_data'

    # 创建目标目录
    os.makedirs(os.path.join(train_data_path, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(train_data_path, 'Negative'), exist_ok=True)
    os.makedirs(os.path.join(val_data_path, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(val_data_path, 'Negative'), exist_ok=True)
    os.makedirs(os.path.join(test_data_path, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(test_data_path, 'Negative'), exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg',  '.png'))]

    # 随机打乱文件顺序
    random.shuffle(image_files)

    # 计算训练集、验证集和测试集的分割点
    total_files = len(image_files)
    train_index = int(total_files * 0.7)
    val_index = int(total_files * 0.9)

    # 分割训练集、验证集和测试集
    train_files = image_files[:train_index]
    val_files = image_files[train_index:val_index]
    test_files = image_files[val_index:]

    # 将文件移动到相应的目录
    with tqdm(total=len(train_files), desc="Processing") as pbar:
        for file_name in train_files:
            pbar.update(1)
            # 去除_rotated_{angle}{ext}后缀
            base_name = file_name.split('_rotated_')[0]
            # 查询Lab Status
            row = deduplicated_data[(deduplicated_data['FileName'] == base_name+'.jpg') | (deduplicated_data['FileName'] == base_name+'.png')]
            if not row.empty:
                lab_status = row['Lab Status'].values[0]
                if lab_status == 'Positive ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(train_data_path, 'Positive', file_name))
                elif lab_status == 'Negative ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(train_data_path, 'Negative', file_name))
            else:
                print(f"未找到文件名: {base_name} 对应的行")

    with tqdm(total=len(val_files), desc="Processing") as pbar:
        for file_name in val_files:
            pbar.update(1)
            # 去除_rotated_{angle}{ext}后缀
            base_name = file_name.split('_rotated_')[0]
            # 查询Lab Status
            row = deduplicated_data[(deduplicated_data['FileName'] == base_name+'.jpg') | (deduplicated_data['FileName'] == base_name+'.png')]
            if not row.empty:
                lab_status = row['Lab Status'].values[0]
                if lab_status == 'Positive ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(val_data_path, 'Positive', file_name))
                elif lab_status == 'Negative ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(val_data_path, 'Negative', file_name))
            else:
                print(f"未找到文件名: {base_name} 对应的行")

    with tqdm(total=len(test_files), desc="Processing") as pbar:
        for file_name in test_files:
            pbar.update(1)
            # 去除_rotated_{angle}{ext}后缀
            base_name = file_name.split('_rotated_')[0]
            # 查询Lab Status
            row = deduplicated_data[(deduplicated_data['FileName'] == base_name+'.jpg') | (deduplicated_data['FileName'] == base_name+'.png')]
            if not row.empty:
                lab_status = row['Lab Status'].values[0]
                if lab_status == 'Positive ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(test_data_path, 'Positive', file_name))
                elif lab_status == 'Negative ID':
                    shutil.copy(os.path.join(source_dir, file_name), os.path.join(test_data_path, 'Negative', file_name))
            else:
                print(f"未找到文件名: {base_name} 对应的行")

def for_image():
    # data_Augmentation()
    split_train_test()

# def for_latlng(lat, lng):
#     return lat, lng

# def for_date(date):
#     date = pd.to_datetime(date)
#     date = date.dt.strftime('%Y-%m-%d')
#     return date

def merge():
    # 读取 CSV 文件
    data_csv = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\2021_MCM_Problem_C_Data\2021MCMProblemC_DataSet.csv')
    images_csv = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\2021_MCM_Problem_C_Data\2021MCM_ProblemC_ Images_by_GlobalID.csv')

    # 合并数据
    merged_data = pd.merge(data_csv, images_csv, on='GlobalID', how='left')

    # 保存合并后的数据
    merged_data.to_csv(r'D:\document\MCM\week2\第二周小测数据\merged_data.csv', index=False)

def deduplicate():
    # 读取合并后的数据
    merged_data = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\merged_data.csv')

    # 统计删除的行数
    initial_count = len(merged_data)

    # 1. 删除FileName为空的行
    merged_data = merged_data[merged_data['FileName'].notna()]

    # 2. 删除FileName对应在指定目录下没有相应文件的行
    merged_data = merged_data[merged_data['FileName'].apply(
        lambda x: all(os.path.isfile(os.path.join(r'D:\document\MCM\week2\第二周小测图片文件\2021MCM_ProblemC_Files', fname.strip())) for fname in x.split(','))
    )]
    deleted_files_count = initial_count - len(merged_data)
    initial_count = len(merged_data)

    # 3. 删除Detection Date/Submission Date为空的行
    merged_data = merged_data.dropna(subset=['Detection Date', 'Submission Date'])
    deleted_dates_count = initial_count - len(merged_data)
    initial_count = len(merged_data)

    # 4. 删除Latitude/Longitude为空的行
    merged_data = merged_data.dropna(subset=['Latitude', 'Longitude'])
    deleted_coordinates_count = initial_count - len(merged_data)
    initial_count = len(merged_data)

    # 5. 删除Lab Status为Unprocessed或者Unverified的行
    merged_data = merged_data[~merged_data['Lab Status'].isin(['Unprocessed', 'Unverified'])]
    deleted_status_count = initial_count - len(merged_data)

    # 保存去重后的数据
    merged_data.to_csv(r'D:\document\MCM\week2\第二周小测数据\deduplicated_data.csv', index=False)

    # 输出统计数据
    print(f"删除的行数：")
    print(f"1. FileName为空的行数: {initial_count - len(merged_data)}")
    print(f"2. 文件不存在的行数: {deleted_files_count}")
    print(f"3. Detection Date/Submission Date为空的行数: {deleted_dates_count}")
    print(f"4. Latitude/Longitude为空的行数: {deleted_coordinates_count}")
    print(f"5. Lab Status为Unprocessed或Unverified的行数: {deleted_status_count}")
    print(f"剩余数据行数: {len(merged_data)}")

def default_prework():
    merge()
    deduplicate()

if __name__ == "__main__":
    # pass
    # default_prework()
    for_image()
