# root = 'tests/data/eval_dataset_v1'
root = 'data/eval_dataset_v1/det'

train = dict(
    type='IcdarDataset',
    ann_file=f'{root}/instances_train.json',
    img_prefix=f'{root}/imgs',
    pipeline=None)

test = dict(
    type='IcdarDataset',
    img_prefix=f'{root}/imgs',
    ann_file=f'{root}/instances_test.json',
    pipeline=None,
    test_mode=True)

train_list = [train]

test_list = [test]
