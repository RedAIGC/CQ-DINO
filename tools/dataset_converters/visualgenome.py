"""
# 创建label map 
import json

object_names = list(open("/mnt/public/usr/sunzhichao/mmdetection/mapping/vg_names.txt", 'r', encoding='utf-8').read().splitlines())

# 创建字典
result_dict = {str(i): name for i, name in enumerate(object_names)}

# 保存为json文件
with open('/mnt/public/usr/sunzhichao/mmdetection/data/VisualGenome/object_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f)
"""


import json
from mmdet.datasets.api_wrappers import COCO

# 读取原始数据
vg_file_name = "/mnt/public/usr/sunzhichao/hf_hub/models--clin1223--GenerateU/train_from_objects.json"
object_names = list(open("/mnt/public/usr/sunzhichao/mmdetection/mapping/vg_names.txt", 'r', encoding='utf-8').read().splitlines())

# 创建名称到ID的映射字典
name_to_id = {name: str(i) for i, name in enumerate(object_names)}

# 读取COCO格式数据
coco = COCO(vg_file_name)
img_ids = coco.get_img_ids()

# 创建新的标注数据
new_annotations = []
new_data = {
    'images': [],
    'annotations': [],
    'categories': [{'id': i, 'name': name} for i, name in enumerate(object_names)]
}

for img_id in img_ids:
    # 获取图像信息
    img_info = coco.load_imgs([img_id])[0]
    
    # img_info['filename'] = img_info.pop('file_name')
    new_data['images'].append(img_info)

    # 获取标注信息
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    annotations = coco.load_anns(ann_ids)
    
    # 更新category_id
    for ann in annotations:
        object_name = ann['object_name']
        # 处理可能的特殊情况
        if object_name in name_to_id:
            ann['category_id'] = int(name_to_id[object_name])
        else:
            print(f"Warning: object name '{object_name}' not found in mapping")
        new_data['annotations'].append(ann)

# 保存新的标注文件
output_file = '/mnt/public/usr/sunzhichao/mmdetection/data/VisualGenome/updated_annotations.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_data, f)

# 打印示例检查
print("Sample image info:", new_data['images'][0])
print("Sample annotation:", new_data['annotations'][0])
