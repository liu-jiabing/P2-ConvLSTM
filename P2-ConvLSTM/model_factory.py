# model_factory.py

import logging
from PhaseAwareLSTM_model import PhaseAwareLSTM
# --- 当你未来添加新模型时，在这里导入它们 ---
# from some_other_model import AnotherModel 

def get_model(model_name, config):
    """
    模型工厂函数。

    根据提供的模型名称，初始化并返回相应的模型实例。

    参数:
        model_name (str): 要创建的模型的名称。
        config (dict): 包含模型参数的配置字典。

    返回:
        torch.nn.Module: 实例化的模型。
    
    异常:
        ValueError: 如果提供的 model_name 无效。
    """
    logging.info(f"Attempting to create model: '{model_name}'")
    
    if model_name.lower() == 'phaseawarelstm':
        # 确保 CONFIG['model'] 被传递给模型
        return PhaseAwareLSTM(config=config)
    
    # --- 当你添加新模型时，在这里添加 elif 分支 ---
    # elif model_name.lower() == 'anothermodel':
    #     return AnotherModel(config=config)
    
    else:
        # 如果找不到指定的模型，则引发错误
        raise ValueError(f"Model '{model_name}' not recognized. Available models: ['PhaseAwareLSTM']")