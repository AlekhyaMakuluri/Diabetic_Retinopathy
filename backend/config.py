"""Training configuration"""

config = {
    # Data loading
    'batch_size': 16,
    'num_workers': 4,
    
    # Model parameters
    'num_classes': 5,
    'hidden_channels': 768,
    
    # Training parameters
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'weight_decay': 2e-4,
    'ssl_weight': 0.2,
    'early_stopping_patience': 30,
    'log_interval': 5,
    
    # Data augmentation
    'augmentation': {
        'train': {
            'resize': (256, 256),
            'random_crop': (224, 224),
            'random_horizontal_flip': True,
            'random_vertical_flip': True,
            'random_rotation': 15,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'random_affine': {
                'degrees': 0,
                'translate': (0.1, 0.1),
                'scale': (0.9, 1.1)
            }
        },
        'val': {
            'resize': (224, 224),
            'center_crop': (224, 224)
        }
    },
    
    # Data paths
    'data_dir': 'backend/data',
    'train_labels': 'backend/data/train_labels.csv',
    'val_labels': 'backend/data/val_labels.csv',
    'test_labels': 'backend/data/test_labels.csv'
} 