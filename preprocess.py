# /dehazyDeepFusionNetwork/preprocess.py

import os
from sklearn.model_selection import train_test_split

def save_file(data, file_name):
    save_path = os.path.join('D:\experiment_data\small_data', file_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'w') as file:
        for index, i in enumerate(data):
            file.write(i)
            if index < len(data)-1:
                file.write('\n')

# 生成检索目录
# 9_98_1 clear_t_浓度
def creat_retrieval_catalog():
    # 处理数据路径信息
    base_path = 'D:\experiment_data\small_data'
    hazy_dir_path = os.path.join(base_path, 'hazy')
    clear_dir_path = os.path.join(base_path, 'clear')
    t_dir_path = os.path.join(base_path, 't')
    # 必要数据生成和获取
    hazy = os.listdir(hazy_dir_path)
    catalog = []
    # 数据获取并改为方便数据集读取格式
    for i in hazy:
        tmp1 = i.split('.')  # 获得文件名和后缀
        suffix = tmp1[1]  # 存储后缀名
        paras = tmp1[0].split('_')  # 获得图片关联信息

        hazy_path = os.path.join(hazy_dir_path, i)  # 单有雾图绝对路径
        clear_path = os.path.join(clear_dir_path, f'{paras[0]}.{suffix}')# 清晰图绝对路径
        t_path = os.path.join(t_dir_path, f'{paras[1]}.{suffix}')  # 转换图绝对路径

        data_joint = f'{hazy_path}|{clear_path}|{t_path}'  # 拼接数据信息
        catalog.append(data_joint)  # 数据存入
    print(f'There are {len(catalog)} images in hazy.')
    train_catalog, test_catalog = train_test_split(catalog, test_size=0.2, random_state=21)
    print("Using {} images for training, {} images for testing.".format(len(train_catalog), len(test_catalog)))
    # 存入数据
    save_file(train_catalog,'train_catalog.txt')
    save_file(test_catalog, 'test_catalog.txt')

if __name__ == '__main__':
    creat_retrieval_catalog()
