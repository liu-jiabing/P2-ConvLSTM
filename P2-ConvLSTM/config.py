# =====================================================================================
# File: config.py (FOR DYNAMIC CORRELATION MODEL)
# =====================================================================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 预训练配置 ---
PRETRAIN_CONFIG = {
    'data': {
        'root_dir': "/scratch/jiabing/data/",
        'stats_path': "/home/jiabing/methods/multi-5/global_stats.npz",
        'variable_mapping': { 't2m':"2m_temperature", 'tcc':"total_cloud_cover", 'tp':"total_precipitation", 'v10':"10m_v_component_of_wind", 'u10':"10m_u_component_of_wind" },
        'variables': ['t2m', 'tcc', 'tp', 'v10', 'u10'],
        'resolution': "5.625deg",
        'lat': 32, 'lon': 64,
        # <<< 关键: 预训练现在也需要序列输入 >>>
        'sequence_params': {'input_len': 12},
        'splits': {'train_years': list(range(2010, 2019))},
        'loader_params': {'batch_size': 32, 'num_workers': 4, 'pin_memory': True, 'shuffle': True},
    },
    'model': {
        'name': 'MaskedAutoencoderForDynamicCorrelation',
        # --- Encoder (DynamicCorrelationModule) 的参数 ---
        'input_channels': 5,
        'hidden_dim': 128,
        'num_interaction_blocks': 3,
        'num_st_blocks': 2,
        # --- MAE 特有的参数 ---
        'decoder_dim': 128,
        'masking_ratio': 0.50,
        'patch_size': 4,
    },
    'training': {
        'device': DEVICE,
        'epochs': 200,
        'learning_rate': 1.5e-4,
        'optimizer': 'AdamW',
        'scheduler': {'name': 'CosineAnnealingLR', 'params': {'T_max': 200, 'eta_min': 1e-6}},
        'save_path': '/home/jiabing/methods/multi-5/multi_2time-4/pretrain/M=0_50/pretrained_dynamic_encoder.pth'
    }
}

# --- 微调配置 ---
FINETUNE_CONFIG = {
    'data': {
        'root_dir': "/scratch/jiabing/data/",
        'stats_path': "/home/jiabing/methods/multi-5/global_stats.npz",
        'variable_mapping': { 't2m':"2m_temperature", 'tcc':"total_cloud_cover", 'tp':"total_precipitation", 'v10':"10m_v_component_of_wind", 'u10':"10m_u_component_of_wind" },
        'variables': ['t2m', 'tcc', 'tp', 'v10', 'u10'],
        'resolution': "5.625deg",
        'lat': 32, 'lon': 64,
        'sequence_params': {'input_len': 12, 'output_len': 12, 'target_variable': 'tp'},
        'splits': {'train_years': list(range(2010, 2016)), 'val_years': list(range(2016, 2017)), 'test_years': list(range(2017, 2019))},
        'loader_params': {'batch_size': 4, 'num_workers': 4, 'pin_memory': True, 'shuffle': True},
    },
    'model': {
        'name': 'PhaseAwareLSTM',
        'patch_size': 1,
        # --- DynamicCorrelationModule 的参数 ---
        'encoder_hidden_dim': 128,
        'num_interaction_blocks': PRETRAIN_CONFIG['model']['num_interaction_blocks'],
        'num_st_blocks': PRETRAIN_CONFIG['model']['num_st_blocks'],
        # --- ConvLSTM 的参数 ---
        'num_hidden': [128,128,128,128,128,128,128,128],
        'layer_norm': True,
        'dropout': 0.05,
        'num_time_features': 0, 
        'period_recognizer_top_k': 2,
        'pretrained_encoder_path': PRETRAIN_CONFIG['training']['save_path'],
        'loss_weights': {'mae_weight': 0.15, 'period_weight': 0.05},
    },
    'training': {
        'device': DEVICE,
        'epochs': 50,
        'learning_rate': 1e-4,
        'optimizer': 'AdamW',
        'scheduler': {'name': 'CosineAnnealingLR', 'params': {'T_max': 50, 'eta_min': 1e-6}},
        'save_dir': './results'
    }
}