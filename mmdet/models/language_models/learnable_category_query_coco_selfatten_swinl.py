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


class LabelSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_classes, embed_dim)
        Returns:
            Tensor of shape (num_classes, embed_dim)
        """
        x = x.unsqueeze(1).permute(1, 0, 2)
        
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.norm(x + attn_output)
        
        return attn_output.squeeze(0)



@MODELS.register_module()
class LearnableCategoryQuerySelfAttentionCOCOSwinL(BaseModel):

    def __init__(self,
                 q2l_config_name='/mnt/public/usr/sunzhichao/mmdetection/q2l_config.json',
                 text_encoder_type='bert-base-uncased',
                 embedding_dir='/mnt/public/usr/sunzhichao/mmdetection/v3det_clip_embeddings.pth',
                 ):
        r""" Tag2Text inference module, both captioning and tagging are included.
        Tag2Text is an efficient and controllable vision-language pre-training framework.
        Described in the paper "Tag2Text: Guiding Vision-Language Model via Image Tagging" https://arxiv.org/abs/2303.05657

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
        """
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

        self.label_attn = LabelSelfAttention(embed_dim=768, num_heads=8)

        self.clean_category_list = [clean_name(item) for item in self.tag_list]

        self.wordvec_proj = nn.Linear(768, 768)
        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()
        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.image_proj = nn.Linear(1536, 512)


    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention
    

    def forward(self, image_embeds):
        """
        call function as forward

        Args:
            image: type: torch.Tensor  shape: batch_size * 3 * 384 * 384
            caption: type: list[string]  len: batch_size
            tag: type: torch.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0

        Returns:
            loss: type: torch.Tensor
        """
        if torch.isnan(self.label_embed).any():
            print("org_ label_embed contain nan")
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))
        label_embed = self.label_attn(label_embed)  # (num_classes, 768)


        B, C, H, W = image_embeds.shape
        image_embeds = image_embeds.view(B, C, -1)  # B C L
        cls_image_embeds = self.avgpool(image_embeds)  # B C 1 
        image_embeds = torch.cat([cls_image_embeds, image_embeds], dim=2).transpose(1, 2) # B L+1 C

        image_embeds = self.image_proj(image_embeds)


        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image_embeds.device)


        label_embed = label_embed.unsqueeze(0).repeat(B, 1, 1)

        if torch.isnan(label_embed).any():
            print("label_embed contain nan")
        if torch.isnan(image_embeds).any():
            print("image_embeds contain nan")


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
            
        logits = self.fc(tagging_embed[0]).squeeze(-1)

        top_values, top_indices = torch.topk(logits, k=30, dim=-1)


        tag_input = []
        tag_indexes = top_indices.tolist()
        for b in range(bs):
            index = tag_indexes[b]
            
            token = self.tag_list[index]

            tag_input.append(token)
            
        tag_output = tag_input



        return tag_output, tag_indexes, logits