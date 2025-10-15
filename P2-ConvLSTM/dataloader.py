# =====================================================================================
# File: dataloader.py (MODIFIED for DynamicCorrelationModule)
# =====================================================================================
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from config import FINETUNE_CONFIG, PRETRAIN_CONFIG

# =====================================================================================
# 1. 用于预训练 (MAE) 的数据集类 - 已修改为输出序列
# =====================================================================================
class PretrainWeatherDataset(Dataset):
    """
    为 DynamicCorrelationModule 的 MAE 预训练定制的数据集。
    - 输出一个长度为 input_len 的序列。
    - 不生成时间特征和未来标签。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.years = config['data']['splits']['train_years']
        # <<< NEW: 预训练现在也需要序列长度 >>>
        self.input_len = config['data']['sequence_params']['input_len']
        
        self._load_data()
        self._load_statistics()

    def _load_statistics(self):
        stats_path = self.config['data']['stats_path']
        stats = np.load(stats_path)
        self.mean = torch.from_numpy(stats['mean']).float()
        self.std = torch.from_numpy(stats['std']).float()
        self.std[self.std == 0] = 1.0
        # 调整形状以匹配序列 [T, C, H, W]
        self.mean_reshaped = self.mean.view(1, -1, 1, 1)
        self.std_reshaped = self.std.view(1, -1, 1, 1)
        print("已为 'pretrain' 数据集加载标准化统计数据。")

    def _load_data(self):
        print(f"Loading data for pre-training, years: {self.years}...")
        all_data_arrays = []
        # 加载所有在config中定义的变量
        for var in self.config['data']['variables']:
            var_map_name = self.config['data']['variable_mapping'].get(var, var)
            file_paths = [os.path.join(self.config['data']['root_dir'], var_map_name, f"{var_map_name}_{year}_{self.config['data']['resolution']}.nc") for year in self.years]
            valid_paths = [p for p in file_paths if os.path.exists(p)]
            if not valid_paths: continue
            var_dataset = xr.open_mfdataset(valid_paths, concat_dim="time", combine="nested", engine='netcdf4')
            all_data_arrays.append(var_dataset[var].fillna(0))
        
        phys_data = xr.concat(all_data_arrays, dim="variable").rename({'variable': 'channel'})
        self.data = phys_data.transpose('time', 'channel', 'lat', 'lon').values
        print(f"Pre-train data loaded. Shape: {self.data.shape}")

    def __len__(self):
        # <<< MODIFIED: 长度现在取决于序列长度 >>>
        return self.data.shape[0] - self.input_len + 1

    def __getitem__(self, idx):
        # <<< MODIFIED: 返回一个序列而不是单个样本 >>>
        start_idx = idx
        end_idx = idx + self.input_len
        
        sequence = self.data[start_idx:end_idx]
        sequence_tensor = torch.from_numpy(sequence).float()
        
        # 标准化整个序列
        sequence_normalized = (sequence_tensor - self.mean_reshaped) / self.std_reshaped
        
        return sequence_normalized

# =====================================================================================
# 2. 用于微调 (Seq2Seq 预测) 的数据集类 - 保持不变
# =====================================================================================
class FinetuneWeatherDataset(Dataset):
    def __init__(self, config, years, mode):
        super().__init__()
        self.config, self.years, self.mode = config, years, mode
        self.input_len = config['data']['sequence_params']['input_len']
        self.output_len = config['data']['sequence_params']['output_len']
        self.total_seq_len = self.input_len + self.output_len
        all_vars = config['data']['variables']
        target_var = config['data']['sequence_params']['target_variable']
        self.target_var_idx = all_vars.index(target_var)
        self.num_phys_vars = len(all_vars)
        self.num_time_features = config['model'].get('num_time_features', 0)
        self._load_data()
        self._load_statistics()

    def _load_statistics(self):
        stats = np.load(self.config['data']['stats_path'])
        self.mean = torch.from_numpy(stats['mean']).float()
        self.std = torch.from_numpy(stats['std']).float()
        self.std[self.std == 0] = 1.0
        self.mean_reshaped = self.mean.view(1, -1, 1, 1)
        self.std_reshaped = self.std.view(1, -1, 1, 1)

    def _load_data(self):
        all_data_arrays = []
        for var in self.config['data']['variables']:
            var_map_name = self.config['data']['variable_mapping'].get(var, var)
            file_paths = [os.path.join(self.config['data']['root_dir'], var_map_name, f"{var_map_name}_{year}_{self.config['data']['resolution']}.nc") for year in self.years]
            valid_paths = [p for p in file_paths if os.path.exists(p)]
            if not valid_paths: continue
            var_dataset = xr.open_mfdataset(valid_paths, concat_dim="time", combine="nested", engine='netcdf4')
            all_data_arrays.append(var_dataset[var].fillna(0))
        
        phys_data = xr.concat(all_data_arrays, dim="variable").rename({'variable': 'channel'})
        
        if self.num_time_features > 0:
            times = phys_data.coords['time']
            day_sin = np.sin(2*np.pi*times.dt.dayofyear/366.0); day_cos = np.cos(2*np.pi*times.dt.dayofyear/366.0)
            hour_sin = np.sin(2*np.pi*times.dt.hour/24.0); hour_cos = np.cos(2*np.pi*times.dt.hour/24.0)
            time_features_list = []
            for feat in [day_sin, day_cos, hour_sin, hour_cos]:
                feat_da = xr.DataArray(feat.values, dims=['time'], coords={'time':times}).expand_dims(dim={'lat':phys_data.lat,'lon':phys_data.lon},axis=[1,2])
                time_features_list.append(feat_da)
            phys_data = xr.concat([phys_data] + time_features_list, dim="channel")

        self.data = phys_data.transpose('time', 'channel', 'lat', 'lon').values

    def __len__(self):
        return len(self.data) - self.total_seq_len + 1

    def __getitem__(self, idx):
        start, mid, end = idx, idx + self.input_len, idx + self.total_seq_len
        input_seq, target_full_seq = self.data[start:mid], self.data[mid:end]
        
        target_seq = target_full_seq[:, self.target_var_idx:self.target_var_idx+1]
        future_time_features = target_full_seq[:, self.num_phys_vars:]
        
        input_tensor = torch.from_numpy(input_seq).float()
        target_tensor = torch.from_numpy(target_seq).float()
        future_tf_tensor = torch.from_numpy(future_time_features).float()
        
        input_phys = (input_tensor[:, :self.num_phys_vars] - self.mean_reshaped) / self.std_reshaped
        input_tensor = torch.cat([input_phys, input_tensor[:, self.num_phys_vars:]], dim=1)
        
        target_mean = self.mean[self.target_var_idx].view(1, 1, 1, 1)
        target_std = self.std[self.target_var_idx].view(1, 1, 1, 1)
        target_tensor = (target_tensor - target_mean) / target_std
        
        return input_tensor, target_tensor, future_tf_tensor

    def inverse_transform(self, tensor):
        mean = self.mean[self.target_var_idx]; std = self.std[self.target_var_idx]
        return tensor * std.to(tensor.device) + mean.to(tensor.device)

# =====================================================================================
# 3. Dataloader 获取函数 - 保持不变
# =====================================================================================
def get_pretrain_dataloader():
    """只返回用于MAE预训练的dataloader"""
    dataset = PretrainWeatherDataset(PRETRAIN_CONFIG)
    params = PRETRAIN_CONFIG['data']['loader_params']
    return DataLoader(dataset, **params)

def get_finetune_dataloaders():
    """返回用于训练、验证、测试的微调Dataloaders"""
    cfg = FINETUNE_CONFIG
    train_ds = FinetuneWeatherDataset(cfg, cfg['data']['splits']['train_years'], 'train')
    val_ds = FinetuneWeatherDataset(cfg, cfg['data']['splits']['val_years'], 'val')
    test_ds = FinetuneWeatherDataset(cfg, cfg['data']['splits']['test_years'], 'test')
    
    params = cfg['data']['loader_params']
    
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=params['shuffle'], num_workers=params['num_workers'], pin_memory=params['pin_memory'])
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=params['pin_memory'])
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=params['pin_memory'])
    
    return train_loader, val_loader, test_loader, val_ds