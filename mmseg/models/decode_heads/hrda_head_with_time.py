# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import torch
from torch.nn import functional as F

from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead

from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@HEADS.register_module()
class HRDAHeadWithTime(BaseDecodeHead):

    def __init__(self,
                 
                 num_feature_levels,
                 encoder,
                 positional_encoding,

                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHeadWithTime, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention


        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs, times):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            # crop_seg_logits = self.head(features)
            crop_seg_logits = self.forward_with_time(features, times)
            
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            # return self.head(inp)
            return self.forward_with_time(inp, times)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward_with_time(self, inputs, times):
        mlvl_feats = inputs[-self.num_feature_levels:]
        
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        out = self.conv_seg(memory)
        return out
    
    def forward(self, inputs, times):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        # lr_seg = self.head(lr_inp)
        lr_seg = self.forward_with_time(lr_inp, times)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_seg = self.decode_hr(hr_inp, batch_size, times)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHeadWithTime, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHeadWithTime, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            if self.debug:
                self.debug_output['Cropped GT'] = \
                    cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHeadWithTime, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHeadWithTime, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        if self.debug:
            self.debug_output['GT'] = \
                seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss
