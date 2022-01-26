# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple


def init_weights(self):
    # logger = get_root_logger()
    if self.init_cfg is None:
        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, 1.0)
    else:
        assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                              f'specify `Pretrained` in ' \
                                              f'`init_cfg` in ' \
                                              f'{self.__class__.__name__} '
        ckpt = _load_checkpoint(self.init_cfg.checkpoint, map_location='cpu')
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        if self.convert_weights:
            # supported loading weight from original repo,
            _state_dict = swin_converter(_state_dict)

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                state_dict[k[9:]] = v

        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # reshape absolute position embedding
        if state_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = state_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.size()
            N2, C2, H, W = self.absolute_pos_embed.size()
            if N1 != N2 or C1 != C2 or L != H * W:
                logger.warning('Error in loading absolute_pos_embed, pass')
            else:
                state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                    N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

        # interpolate position bias table if needed
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if 'relative_position_bias_table' in k
        ]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = state_dict[table_key]
            table_current = self.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if nH1 != nH2:
                logger.warning(f'Error in loading {table_key}, pass')
            elif L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(
                    nH2, L2).permute(1, 0).contiguous()

        # load state_dict
        self.load_state_dict(state_dict, False)
