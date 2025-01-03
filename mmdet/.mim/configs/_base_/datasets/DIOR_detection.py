dataset_type = 'DIORDataset'
data_root = 'data/DIORcoco/'
classes = ("airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'Annotations/DIOR_trainval.json',
        img_prefix=data_root + 'JPEGImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'Annotations/DIOR_test.json',
        img_prefix=data_root + 'JPEGImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'Annotations/DIOR_test.json',
        img_prefix=data_root + 'JPEGImages',
        pipeline=test_pipeline))
evaluation = dict(interval=12, metric='bbox')
