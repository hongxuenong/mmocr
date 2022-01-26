model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        depths=(2, 2, 18, 2),  #Swin-L
        num_heads=[6, 12, 24, 48],
        window_size=12,
        # pretrained='saved_models/swin_large_patch4_window12_384_22k.pth',
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=False,
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='saved_models/swin_large_patch4_window12_384_22k.pth'),
    ),
    neck=dict(
        type='FPNC', in_channels=[192, 384, 768, 1536], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=1.0, beta=10.0, bbce_loss=True),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)
