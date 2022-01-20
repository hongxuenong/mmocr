_base_ = [
    '../../_base_/runtime_10e.py',
    '../../_base_/schedules/schedule_sgd_1500e.py',
    '../../_base_/det_models/fcenet_r50dcnv2_fpn.py',
    '../../_base_/det_datasets/eval_dataset_v1.py',
    '../../_base_/det_pipelines/fcenet_pipeline.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline_shopee = {{_base_.train_pipeline_shopee}}
test_pipeline_shopee = {{_base_.test_pipeline_shopee}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_shopee),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_shopee),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_shopee))

evaluation = dict(interval=5, metric='hmean-iou')
