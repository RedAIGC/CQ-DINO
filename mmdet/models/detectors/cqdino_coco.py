# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
import random
from ..layers.transformer.cqdino_layers import CQDinoTransformerEncoder
import numpy as np

def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name

def map_numbers(input_list):
    # 用于存储数字与其映射之间的关系
    mapping_dict = {}
    # 结果列表
    mapped_list = []

    for num in input_list:
        if num not in mapping_dict:
            # 生成 1 到 127 之间的随机数，不包括0
            mapping_dict[num] = random.randint(0, 126)
        # 添加映射后的结果到mapped_list中
        mapped_list.append(mapping_dict[num])
    
    return mapped_list

def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_

def create_position_ids(length):
    # Create a tensor of the specified length filled with zeros
    # position_ids = torch.zeros(length, dtype=torch.long)
    
    # # Set every second element to 1 starting from index 2
    # position_ids[2:length:2] = 1

    # # plan 1 
    # position_ids = torch.zeros(length, dtype=torch.long)
    # attention_mask = torch.eye(length)

    # plan 2 
    position_ids = torch.arange(length, dtype=torch.long)
    attention_mask = torch.ones((length, length), dtype=bool)

    return position_ids, attention_mask


# def create_attention_mask(position_ids):

#     new_ids = torch.zeros_like(position_ids)
#     new_ids[0] = 1
#     new_ids[-1] = 1 
#     idxs = torch.nonzero(new_ids)
#     attention_mask = torch.eye(len(new_ids)).bool()
#     previous_col = 0
#     for i in range(len(idxs)):
#         col = idxs[i]
#         if col  == 0 or col ==len(idxs)-1 :
#             attention_mask[col, col] = True
#         else:
#             attention_mask[previous_col+1:col+1, previous_col+1:col+1] = True

#         previous_col = col

#     return attention_mask
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.default_loss = torch.tensor(10.0)  # 可以根据需要调整这个值



    def forward(self, x, y):
        # 添加输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Input contains NaN or inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e2, neginf=-1e2)
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            return self.default_loss.to(x.device)

        # 限制输入范围，避免sigmoid饱和
        x = torch.clamp(x, min=-88.0, max=88.0)
        y = torch.clamp(y, min=0.0, max=1.0)

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 确保概率值在有效范围内
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0)
        xs_neg = xs_neg.clamp(min=self.eps, max=1.0)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos)
        los_neg = (1 - y) * torch.log(xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            
            # 确保pt的值在有效范围内
            pt = pt.clamp(min=self.eps, max=1.0)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # 最后检查loss是否有效
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Warning: Loss contains NaN or inf values")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e2, neginf=-1e2)

        return -loss.sum()


@MODELS.register_module()
class CQDINOCOCO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        # print("!!!!!", self.language_model_cfg)
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = CQDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        
        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.language_loss_func = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)
        # defalut language dim   768
        self.text_feat_map = nn.Linear(
            768,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        org_feature = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(org_feature)
        return x, org_feature

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        # batch_inputs  == images  torch.Size([4, 3, 966, 1059])
        """
        batch_data_samples: meta information
        scale_factor: (0.6223958333333334, 0.6222222222222222)
        batch_input_shape: (750, 1333)
        ori_shape: (1080, 1920)
        text: 'the white pedestrians ride bicycles on the road.'
        flip: False
        tokens_positive: 
            0: [[0, 3], [4, 9], [10, 21]]
        pad_shape: (672, 1195)
        img_shape: (672, 1195)
        img_path: '/mnt/public/usr/sunzhichao/VisDrone2019/all_image/0000165_02725_d_0000097.jpg'
           
        DATA FIELDS
        ignored_instances: <InstanceData(
            
                META INFORMATION
            
                DATA FIELDS
                bboxes: tensor([], device='cuda:0', size=(0, 4))
                labels: tensor([], device='cuda:0', dtype=torch.int64)
            ) at 0x7f89b4c45f60>
        gt_instances: <InstanceData(
            
                META INFORMATION
            
                DATA FIELDS
                bboxes: tensor([[637.9557, 388.8889, 644.8021, 403.8222],
                            [626.7526, 393.2444, 633.5989, 408.1778],
                            [637.3333, 397.6000, 643.5573, 410.6667],
                            [521.5677, 419.3778, 535.2604, 449.2444]], device='cuda:0')
                labels: tensor([0, 0, 0, 0], device='cuda:0')
            ) at 0x7f89b4c46fb0>
        """

        batchsize = len(batch_data_samples)
        if self.use_autocast:
            with autocast(enabled=True):
                neck_visual_features, visual_features = self.extract_feat(batch_inputs)
        else:
            neck_visual_features, visual_features = self.extract_feat(batch_inputs)

        tagging_embed = self.language_model(visual_features[-1])

        tag_output, tag_indexes, logits = self.language_model.generate(visual_features[-1], tagging_embed)  # tag words 从0开始

        target_len = 30
        text_token_mask = []
        new_tagging_embed = []

        tag_logits = []
        for b in range(batchsize):
            index = tag_indexes[b]
            t_logits = logits[b][index]
            tag_logits.append(t_logits)

        tag_logits = torch.stack(tag_logits, dim=0)

        for embed, tag_index in zip(tagging_embed[0], tag_indexes):

            new_embed = embed[tag_index]
            org_len = new_embed.shape[0]
            pad_len = target_len - new_embed.shape[0]
            tag_attention_mask = torch.zeros(target_len, device=embed.device)
            if pad_len > 0:
                used_mask = torch.zeros(len(embed), dtype=torch.bool, device=embed.device)
                used_mask[tag_index] = True
                unused_indices = torch.where(~used_mask)[0]
                pad_indices = unused_indices[torch.randperm(len(unused_indices))[:pad_len]]
                padding = embed[pad_indices]
                new_embed = torch.cat([new_embed, padding], dim=0)
                tag_attention_mask[:org_len] = 1

            else:
                new_embed = new_embed[:target_len]
                tag_attention_mask[:target_len] = 1


            text_token_mask.append(tag_attention_mask)
            new_tagging_embed.append(new_embed)

        text_token_mask = torch.stack(text_token_mask).to(embed.device).bool()
        new_tagging_embed = torch.stack(new_tagging_embed).to(embed.device)


        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples]
        
        positive_maps = []
        image_categories = torch.zeros_like(logits).to(logits.device)

        for text_prompt, gt_label, tag_index, image_category in zip(text_prompts, gt_labels, tag_indexes, image_categories):
            if len(tag_index) > target_len:
                tag_index = tag_index[:target_len]
            class_names = text_prompt.split(". ")[:-1]
            class_indexes = []
            common_index = []

            for class_name in class_names:
                class_index = self.language_model.clean_category_list.index(class_name)
                image_category[class_index] = 1

                if class_index in tag_index:
                    index_position = np.where(np.array(tag_index) == class_index)[0][0] 
                    common_index.append(index_position)
                else:
                    common_index.append(-1)

                class_indexes.append(class_index)

            image_filename = [
                data_samples.img_path for data_samples in batch_data_samples
            ]
            
            positive_map_per_image = []
            for i in range(len(gt_label)):
                positive_map = torch.zeros(target_len)
                word_position = gt_label[i]
                j = common_index[word_position]
                if j == -1:
                    pass
                else:
                    positive_map[j] = 1
                positive_map_per_image.append(positive_map)
            if len(positive_map_per_image) != 0:
                temp_positive_map_per_image = torch.stack(positive_map_per_image)
            else:
                temp_positive_map_per_image = torch.empty((0, target_len))
            positive_maps.append(temp_positive_map_per_image)


        if self.text_feat_map is not None:
            new_tagging_embed = self.text_feat_map(new_tagging_embed)


        position_ids, attention_mask = create_position_ids(new_tagging_embed.shape[1])

        position_ids = position_ids.unsqueeze(0).expand(batchsize, -1).to(batch_inputs.device)
        attention_mask = attention_mask.unsqueeze(0).expand(batchsize, -1, -1).to(batch_inputs.device)


        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        text_dict = {
            'embedded': new_tagging_embed,
            'position_ids': position_ids,
            'text_token_mask': text_token_mask,
            'masks': attention_mask
        }
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        head_inputs_dict = self.forward_transformer(neck_visual_features, text_dict,
                                                    batch_data_samples)
        

        language_loss = self.language_loss_func(logits, image_categories)


        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        losses["language_loss"] = 0.1 * language_loss
        

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):

        batchsize = len(batch_data_samples)
        if self.use_autocast:
            with autocast(enabled=True):
                neck_visual_features, visual_features = self.extract_feat(batch_inputs)
        else:
            neck_visual_features, visual_features = self.extract_feat(batch_inputs)

        tagging_embed = self.language_model(visual_features[-1])

        tag_output, tag_indexes, logits = self.language_model.generate(visual_features[-1], tagging_embed) 

        num_class = self.language_model.num_class
        tag_logits = []
        for b in range(batchsize):
            index = tag_indexes[b]
            t_logits = logits[b][index]
            tag_logits.append(t_logits)

        tag_logits = torch.stack(tag_logits, dim=0)

        target_len = 30 
        text_token_mask = []
        new_tagging_embed = []


        for embed, tag_index in zip(tagging_embed[0], tag_indexes):
            if len(tag_index) >= target_len:
                tag_index = tag_index[:target_len]
            new_embed = embed[tag_index]
            org_len = new_embed.shape[0]
            pad_len = target_len - new_embed.shape[0]
            tag_attention_mask = torch.zeros(target_len, device=embed.device)
            if pad_len > 0:
                used_mask = torch.zeros(len(embed), dtype=torch.bool, device=embed.device)
                used_mask[tag_index] = True
                unused_indices = torch.where(~used_mask)[0]
                pad_indices = unused_indices[torch.randperm(len(unused_indices))[:pad_len]]
                padding = embed[pad_indices]
                new_embed = torch.cat([new_embed, padding], dim=0)
                tag_attention_mask[:org_len] = 1

            else:
                new_embed = new_embed[:target_len]
                tag_attention_mask[:target_len] = 1


            text_token_mask.append(tag_attention_mask)
            new_tagging_embed.append(new_embed)

        text_token_mask = torch.stack(text_token_mask).to(embed.device).bool()
        new_tagging_embed = torch.stack(new_tagging_embed).to(embed.device)


        entities = tag_output
        token_positive_maps = []
        for tag_index in tag_indexes:
            token_positive_maps.append([{i+1:[int(i)] for i in range(len(tag_index))}])


        if self.text_feat_map is not None:
            new_tagging_embed = self.text_feat_map(new_tagging_embed)

        position_ids, attention_mask = create_position_ids(new_tagging_embed.shape[1])
        position_ids = position_ids.unsqueeze(0).expand(batchsize, -1).to(batch_inputs.device)
        attention_mask = attention_mask.unsqueeze(0).expand(batchsize, -1, -1).to(batch_inputs.device)

        text_dict = {
            'embedded': new_tagging_embed,
            'position_ids': position_ids,
            'text_token_mask': text_token_mask,
            'masks': attention_mask
        }        

        num_class = self.language_model.num_class

        positive_maps = []
        for tag_index_pre_image in tag_indexes:
            if len(tag_index_pre_image) >= target_len:
                tag_index_pre_image = tag_index_pre_image[:target_len]
            positive_map_per_image = []
            for tag_index in tag_index_pre_image:
                positive_map = torch.zeros(num_class)
                positive_map[tag_index] = 1
                positive_map_per_image.append(positive_map)
            
            if len(positive_map_per_image) != 0:
                temp_positive_map_per_image = torch.stack(positive_map_per_image)
            else:
                temp_positive_map_per_image = torch.empty((0, num_class))

            positive_maps.append(temp_positive_map_per_image)


        is_rec_tasks = []
        for i, data_samples in enumerate(batch_data_samples):

            data_samples.token_positive_map = token_positive_maps[i][0]
            is_rec_tasks.append(False)

        head_inputs_dict = self.forward_transformer(neck_visual_features, text_dict, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        
        for data_sample, pred_instances, entity, is_rec_task, weight in zip(
                batch_data_samples, results_list, entities, is_rec_tasks, tag_logits):
            if len(pred_instances) > 0:
                label_names = []
                pred_class_index = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                        t_class_index = self.language_model.clean_category_list.index(entity[labels])
                        pred_class_index.append(t_class_index)


                pred_instances.class_labels = torch.tensor(pred_class_index).to(batch_inputs.device)

                pred_instances.label_names = label_names


            data_sample.pred_instances = pred_instances
        return batch_data_samples

