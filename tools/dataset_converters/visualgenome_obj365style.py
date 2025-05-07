import json
from mmdet.datasets.api_wrappers import COCO
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def has_invalid_chars(text):
    # 返回True如果包含非字母和非空格的字符
    return any(not (char.isalpha() or char.isspace()) for char in text)

def extract_nouns(phrase):
    # 对文本进行词性标注
    tokens = word_tokenize(phrase)
    tagged = pos_tag(tokens)
    
    # 提取名词 (NN:名词单数, NNS:名词复数, NNP:专有名词单数, NNPS:专有名词复数)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    
    # 如果找到名词，返回最后一个名词（通常是主要名词）
    if nouns:
        return nouns[-1]
    return phrase  # 如果没有找到名词，返回原始短语

object_names = list(open("/mnt/public/usr/sunzhichao/mmdetection/vg_names_v3.txt", 'r', encoding='utf-8').read().splitlines())
name_to_id = {name: str(i) for i, name in enumerate(object_names)}

def convert_coco_to_custom(coco_json_path, output_path):
    # 读取COCO格式的JSON文件
    coco = COCO(coco_json_path)
    img_ids = coco.get_img_ids()

    with open(output_path, 'w') as f:

        for img_id in img_ids:
            raw_img_info = coco.load_imgs([img_id])[0]
            ann_ids = coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = coco.load_anns(ann_ids)
            new_format = {
                "filename": raw_img_info['file_name'],
                "height": raw_img_info['height'],
                "width": raw_img_info['width'],
                "detection": {
                        "instances": []
                    }
            }
            for ann in raw_ann_info:
                bbox = ann['bbox']
                label = ann['category_id']
                category = ann['object_name']
                if has_invalid_chars(category) is False:
                    noun = extract_nouns(category)
                    if noun in object_names:
                        label = int(name_to_id[noun])    
                        category = noun

                        instance = {
                            "bbox": bbox,
                            "label": label,  # 假设类别ID从1开始，需要减1
                            "category": category
                        }

                        new_format['detection']['instances'].append(instance)

            if len(new_format['detection']['instances']) != 0:
                f.write(json.dumps(new_format) + '\n')



coco_json_path = "/mnt/public/usr/sunzhichao/hf_hub/models--clin1223--GenerateU/train_from_objects.json"
output_path = '/mnt/public/usr/sunzhichao/mmdetection/data/VisualGenome/visualgenome_od_v2.json'


convert_coco_to_custom(coco_json_path, output_path)


