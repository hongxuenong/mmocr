root = '../data/syn+MTWI2018'

train = dict(
    type='IcdarDataset',
    ann_file=f'{root}/instances_training.json',
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
