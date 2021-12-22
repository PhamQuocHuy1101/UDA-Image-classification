global_device = 'cuda:1'
display_step = 100

optim = {
    'lr': 0.001,
    'min_lr': 1e-6,
    'max_iter': 1000,
    'warn_up_step': 100,
    'weight_decay': 1e-4,
    'sup_batch_size': 32,
    'unsup_batch_size': 180,
    'tsa_method': 'exp',
    'unsup_beta': 0.8,
    'unsup_teta': 0.4,
    'unsup_lambda': 1.0
}

model = {
    'n_class': 4,
    'backbone': 'resnet50',
    'n_fc': 2,
    'drop': 0.12,
    'checkpoint': './checkpoint'
}

label_data = {
    'dir_path': './data/data_v1/label',
    'train_path': './data/data_v1/labeled_train.csv',
    'val_path': './data/data_v1/labeled_val.csv',
}
unlabel_data = {
    'dir_path': './data/data_v1/unlabel',
    'train_path': './data/data_v1/unlabeled_train.csv',
    'val_path': './data/data_v1/unlabeled_val.csv',
}
