# =====================================================================================
# File: test.py (ADAPTED for DynamicCorrelationModule)
# =====================================================================================
import os
import argparse
import logging
import time
import torch
import numpy as np
from tqdm import tqdm

# --- 导入我们自己的模块 ---
from config import FINETUNE_CONFIG
from dataloader import get_finetune_dataloaders
from model_factory import get_model
from utils.metrics import MetricsCalculator 

def setup_logging(log_dir, run_name):
    """配置日志系统，用于测试过程。"""
    log_filename = os.path.join(log_dir, f"test_{run_name}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Test log will be saved to {log_filename}")

def main(args):
    # --- 1. 设置 ---
    cfg = FINETUNE_CONFIG
    run_dir = os.path.join(args.output_dir, args.run_name)
    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_checkpoint.pth')
    save_dir = os.path.join(run_dir, 'saved')
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(run_dir, args.run_name)
    
    if not os.path.exists(checkpoint_path):
        logging.error(f"FATAL: Best checkpoint file not found at '{checkpoint_path}'")
        return

    # --- 2. 数据 ---
    logging.info("Loading test dataset...")
    device = cfg['training']['device']
    target_variable = cfg['data']['sequence_params']['target_variable']
    
    _, _, test_loader, test_dataset = get_finetune_dataloaders()
    if not test_loader:
        logging.error("Test dataloader could not be created."); return
    logging.info(f"Test data loaded. Target variable: '{target_variable}'")

    # --- 3. 加载模型 ---
    logging.info(f"Loading best model from: {checkpoint_path}")
    model_name = args.model_name if args.model_name else cfg['model']['name']
    model = get_model(model_name, cfg).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}")
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}"); return
        
    model.eval()

    # --- 4. 初始化 ---
    metrics_calculator = MetricsCalculator(metrics_list=['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr'])
    
    # 新模型的 target_var_idx 可以直接从 config 获取，更稳健
    try:
        target_var_idx = cfg['data']['variables'].index(cfg['data']['sequence_params']['target_variable'])
    except Exception as e:
        logging.error(f"Could not determine target variable index. Error: {e}"); return
    
    all_inputs_std_list = []
    all_preds_std_list = []
    all_trues_std_list = []

    # --- 5. 测试循环 ---
    logging.info("Starting evaluation on the test set...")
    test_pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for x_test, y_test, future_feats_test in test_pbar:
            x_test, y_test = x_test.to(device), y_test.to(device)
            future_feats_test = future_feats_test.to(device) if future_feats_test.numel() > 0 else None
            
            predictions, _ = model(x_test, future_feats_test)

            all_inputs_std_list.append(x_test.cpu().numpy()[:, :, target_var_idx:target_var_idx+1, :, :])
            all_preds_std_list.append(predictions.cpu().numpy())
            all_trues_std_list.append(y_test.cpu().numpy())
            
            predictions_rescaled = test_dataset.inverse_transform(predictions)
            y_test_rescaled = test_dataset.inverse_transform(y_test)
            metrics_calculator.update(predictions_rescaled, y_test_rescaled)

    # --- 6. 保存结果 ---
    final_metrics = metrics_calculator.compute()
    
    logging.info("Concatenating and saving STANDARDIZED sequences to single .npy files...")
    try:
        final_inputs = np.concatenate(all_inputs_std_list, axis=0)
        final_preds = np.concatenate(all_preds_std_list, axis=0)
        final_trues = np.concatenate(all_trues_std_list, axis=0)
        
        logging.info(f"  - Final input shape: {final_inputs.shape}")
        logging.info(f"  - Final pred shape:  {final_preds.shape}")
        logging.info(f"  - Final true shape:  {final_trues.shape}")
        
        np.save(os.path.join(save_dir, 'inputs.npy'), final_inputs)
        np.save(os.path.join(save_dir, 'preds.npy'), final_preds)
        np.save(os.path.join(save_dir, 'trues.npy'), final_trues)
        
    except ValueError as e:
        logging.error(f"Could not concatenate and save sequences. Error: {e}")
    
    # --- 7. 打印报告 ---
    display_metrics = {}
    if target_variable == 'tp':
        logging.info("Formatting metrics for 'tp' (Total Precipitation)...")
        for name, score in final_metrics.items():
            if name == 'mse': display_metrics[f'MSE (1e-7)'] = score * 1e7
            elif name == 'mae': display_metrics[f'MAE (1e-5)'] = score * 1e5
            elif name == 'rmse': display_metrics[f'RMSE (1e-3)'] = score * 1e3
            else: display_metrics[name.upper()] = score
    else:
        for name, score in final_metrics.items(): display_metrics[name.upper()] = score

    logging.info("-" * 50)
    logging.info("Final Test Results (calculated on rescaled data):")
    for name, score in display_metrics.items():
        logging.info(f"  - {name}: {score:.4e}")
    logging.info("-" * 50)
    
    metrics_save_path = os.path.join(save_dir, 'metrics.npy')
    np.save(metrics_save_path, final_metrics)
    
    logging.info(f"Evaluation finished. STANDARDIZED sequences and metrics saved in: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the trained weather prediction model.")
    parser.add_argument('--run_name', type=str, required=True, help="Name of the training run to test.")
    parser.add_argument('--output_dir', type=str, default=FINETUNE_CONFIG['training']['save_dir'], help="Directory where results are saved.")
    parser.add_argument('--model_name', type=str, default=None, help="Name of the model to test. Overrides config.")
    args = parser.parse_args()
    main(args)