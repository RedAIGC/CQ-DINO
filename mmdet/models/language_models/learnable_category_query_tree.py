'''
 * The Tag2Text Model
 * Written by Xinyu Huang
'''
import numpy as np
import torch
import warnings

from torch import nn
from .tagbertmodel import BertConfig, BertModel

from .utils import *

warnings.filterwarnings("ignore")
from mmengine.model import BaseModel
from mmdet.registry import MODELS
import pandas as pd
import re
import csv


def read_mapping_csv(file_name):
    df = pd.read_csv(file_name)
    word_to_index = dict(zip(df['Word_in_B'], df['Index_in_A']))
    return word_to_index


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()

    return name



def csv_to_json(csv_file_path):
    result = {}
    levels = {}
    all_levels = {}
    all_node_info = {}
    max_children_count = 13204
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        next(csv_file)
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            level, id_, category_id, english_name, chinese_name, english_description, \
            chinese_description, parent_id, direct_children, ancestor_category_ids = row
            level = int(level)
            if direct_children == '':
                children_id = []
            else:
                children_id = [int(id) for id in direct_children.split(',')]

            if ancestor_category_ids == '':
                ancestor_id = []
            else:
                ancestor_id = [int(id) for id in ancestor_category_ids.split(',')]

            value = {
                "Level": level,
                "ID": id_,
                "English Name": english_name,
                "Chinese Name": chinese_name,
                "English Description": english_description,
                "Chinese Description": chinese_description,
                "Parent ID": parent_id,
                "Direct Children": children_id,
                "Ancestor Category IDs": ancestor_id,
                "Total Children Count": 0,  # 初始化子节点数量
                "Weight": 0.0  # 初始化权重
            }

            all_node_info[category_id] = value

    def count_all_children(category_id):
        node = all_node_info[category_id]
        direct_children = node["Direct Children"]
        
        if not direct_children:
            return 0
        
        total = len(direct_children)
        for child_id in direct_children:
            total += count_all_children(str(child_id))
        
        all_node_info[category_id]["Total Children Count"] = total
        return total

    for category_id in all_node_info:
        if not all_node_info[category_id]["Parent ID"]: 
            count_all_children(category_id)

    for category_id, node_info in all_node_info.items():
        total_children = node_info["Total Children Count"]
        if max_children_count > 0:
            raw_weight = math.log(1 + total_children + 1) / math.log(1 + max_children_count + 1)
            normalized_weight = 0.4 + (raw_weight * 0.4)
        else:
            normalized_weight = 0.5
        
        all_node_info[category_id]["Weight"] = normalized_weight

    for category_id, value in all_node_info.items():
        if value["Direct Children"]:
            result[category_id] = value

        level = value["Level"]
        if level == 8 or level == 9:
            level = 7

        if level not in levels:
            levels[level] = []
        if value["Direct Children"]:
            levels[level].append(int(category_id))

        if level not in all_levels:
            all_levels[level] = []
        all_levels[level].append(int(category_id))

    return result, levels, all_levels, all_node_info



class TreeFeatureProcessor:
    def __init__(self, tree_info, levels, all_levels, device):

        self.device = device
        self.level_info = {}
        

        for level in levels.keys():
            children_indices = []
            parent_indices = []
            update_weights = [] 
            
            for category_id in levels[level]:
                str_cat_id = str(category_id)

                children = tree_info[str_cat_id]["Direct Children"]
                
                if children:
                    for child_id in children:
                        children_indices.append(child_id)
                        parent_indices.append(int(category_id))

                        
            if children_indices:

                update_indices = [int(cat_id) for cat_id in levels[level]
                                if str(cat_id) in tree_info and
                                tree_info[str(cat_id)]["Direct Children"]]
                
                update_weights = [tree_info[str(idx)]["Weight"] for idx in update_indices]

                self.level_info[level] = {
                    'children_indices': torch.tensor(children_indices, device=device),
                    'parent_indices': torch.tensor(parent_indices, device=device),
                    'update_indices': torch.tensor([int(cat_id) for cat_id in levels[level]
                                                if str(cat_id) in tree_info and
                                                tree_info[str(cat_id)]["Direct Children"]],
                                               device=device),
                    'update_weights': torch.tensor(update_weights, device=device) 

                
                }

    def calculate_features(self, features):

        for level in sorted(self.level_info.keys(), reverse=True):
            info = self.level_info[level]
            
            if len(info['children_indices']) > 0:
                children_features = features[:, info['children_indices'], :]
                
                batch_size, _, dim = features.size()
                scatter_indices = info['parent_indices'].view(1, -1, 1).expand(batch_size, -1, dim)
                
                pooled = torch.zeros_like(features)
                pooled.scatter_add_(1, scatter_indices, children_features)
                
                count = torch.zeros(batch_size, features.size(1), device=self.device)
                count_indices = info['parent_indices'].view(1, -1)
                count.scatter_add_(1, count_indices,
                                 torch.ones(batch_size, len(info['parent_indices']),
                                          device=self.device))
                
                valid_mask = count > 0
                valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, dim)
                count = count.unsqueeze(-1).expand(-1, -1, dim)
                
                pooled[valid_mask] = pooled[valid_mask] / count[valid_mask]
                

                update_indices = info['update_indices']
                weights = info['update_weights']
                
                update_mask = torch.zeros(batch_size, features.size(1), dtype=torch.bool, device=self.device)
                update_mask[:, update_indices] = True
                update_mask = update_mask.unsqueeze(-1).expand(-1, -1, dim)
                update_mask = update_mask & valid_mask
                
                weight_matrix = torch.zeros_like(features, dtype=features.dtype)
                weight_matrix[:, update_indices] = weights.view(1, -1, 1).expand(batch_size, -1, dim)
                features = torch.where(update_mask,
                                    (1 - weight_matrix) * features + weight_matrix * pooled,
                                    features)

        
        return features



@MODELS.register_module()
class LearnableCategoryQueryTree(BaseModel):
    def __init__(self,
                 q2l_config_name,
                 text_encoder_type,
                 embedding_dir,
                 tree_structure,
                 ):

        super().__init__()


        self.tokenizer = init_tokenizer(text_encoder_type)

        q2l_config = BertConfig.from_json_file(q2l_config_name)
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))

        self.label_embed = nn.Parameter(torch.load(embedding_dir, map_location='cpu')['embeddings'].float())

        self.tag_list = np.array(torch.load(embedding_dir, map_location='cpu')['categories'])
        self.num_class = len(self.tag_list)

        self.label_norm = nn.LayerNorm(self.label_embed.size(-1))


        self.tree_info, self.level_info, self.all_levels, self.all_node_info = csv_to_json(tree_structure)

        self.tree_processor = TreeFeatureProcessor(self.tree_info, self.level_info, self.all_levels, "cuda")

        self.clean_category_list = [clean_name(item) for item in self.tag_list]

        self.wordvec_proj = nn.Linear(768, 768)
        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()
        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.image_proj = nn.Linear(1024, 512)


    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention
    
    def forward(self, image_embeds):

        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))


        B, C, H, W = image_embeds.shape
        image_embeds = image_embeds.view(B, C, -1)  # B C L
        cls_image_embeds = self.avgpool(image_embeds)  # B C 1 
        image_embeds = torch.cat([cls_image_embeds, image_embeds], dim=2).transpose(1, 2) # B L+1 C

        image_embeds = self.image_proj(image_embeds)


        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image_embeds.device)


        label_embed = label_embed.unsqueeze(0).repeat(B, 1, 1)



        label_embed = self.tree_processor.calculate_features(label_embed)
  

        label_embed = self.label_norm(label_embed)



        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )


        return tagging_embed


    def generate(self, image_embeds, tagging_embed):
        bs = image_embeds.shape[0]
        device = image_embeds.device

            
        logits = self.fc(tagging_embed[0]).squeeze(-1)


        top_values, top_indices = torch.topk(logits, k=100, dim=-1)


        tag_input = []
        tag_indexes = top_indices.tolist()
        for b in range(bs):
            index = tag_indexes[b]
            
            token = self.tag_list[index]

            tag_input.append(token)
            
        tag_output = tag_input



        return tag_output, tag_indexes, logits