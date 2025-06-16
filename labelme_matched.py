import json
import os
import numpy as np
import cv2
def calculate_iou(shape1, shape2):
    x1, y1, x2, y2 = calculate_point(shape1)
    x3, y3, x4, y4 = calculate_point(shape2)

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_point(shape):
    points = shape['points']
    # 提取所有点的 x 和 y 坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    # 使用 min 和 max 找到左上角点和右下角点
    x1, y1 = min(x_coords), min(y_coords)  # 左上角点
    x2, y2 = max(x_coords), max(y_coords)  # 右下角点
    return x1, y1, x2, y2


def process_json_files(json_path1, json_path2, output_path):
    data1 = load_json(json_path1)
    data2 = load_json(json_path2)
    # 按标签分组
    label_groups = {
        'face': {'original': [], 'new': []},
        'facemask': {'original': [], 'new': []},
        'head': {'original': [], 'new': []},
        'headmask': {'original': [], 'new': []}
    }
    
    for shape in data1['shapes']:
        if shape['label'] in label_groups:
            label_groups[shape['label']]['original'].append(shape)
    
    for shape in data2['shapes']:
        if shape['label'] in label_groups:
            label_groups[shape['label']]['new'].append(shape)
    
    # 匹配和处理
    t=0.5#误差阈值 1-iou
    for label, groups in label_groups.items():
        original_shapes = groups['original']
        new_shapes = groups['new']
        for original_shape in original_shapes[:]:
            min_t=1
            ori_s=None
            new_s=None
            matched=False
            for new_shape in new_shapes:
                iou=1-calculate_iou(original_shape, new_shape)
                if(iou<min_t):
                    min_t=iou
                    ori_s=original_shape
                    new_s=new_shape
                if(min_t<t):
                    matched=True
            if not matched:
                original_shape['group_id'] = 58
            else:
                # 删除最优匹配的标注
                original_shapes.remove(ori_s)
                new_shapes.remove(new_s)

        for new_shape in new_shapes:
            new_shape['group_id'] = 68
        #print(original_shapes)
        #print(new_shapes)
    # 合并 shapes，只保留有 group_id 的
    merged_shapes = []
    for shapes in label_groups.values():
        for shape in shapes['original'] + shapes['new']:
            if shape['group_id']!=None:
                merged_shapes.append(shape)


    data1['shapes'] = merged_shapes
    # 保存到新的 JSON 文件
    save_json(data1, output_path)
    print(f" go-> {output_path}")

def process_directories(dir_path1, dir_path2, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files1 = {os.path.basename(f): f for f in os.listdir(dir_path1) if f.endswith('.json')}
    files2 = files1
    
    for file_name in set(files1.keys()) & set(files2.keys()):
        json_path1 = os.path.join(dir_path1, files1[file_name])
        json_path2 = os.path.join(dir_path2, files2[file_name])
        output_path = os.path.join(output_dir, file_name)
        process_json_files(json_path1, json_path2, output_path)

if __name__ == "__main__":
    dir_path1 = 'D:/tempdata/cvte_va_headdet/1.test_ori_output'  # 原 JSON 文件目录路径
    dir_path2 = 'D:/tempdata/cvte_va_headdet/1.test_ai_output' # 新 JSON 文件目录路径
    output_dir = 'D:/tempdata/cvte_va_headdet/1.test' # 输出 JSON 文件目录路径

    process_directories(dir_path1, dir_path2, output_dir)
