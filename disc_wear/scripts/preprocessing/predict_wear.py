#!/usr/bin/env python3
"""
刀具磨损预测脚本
加载预训练网络对44把刀的磨损值进行预测
参考calcu_friction_energy.py的预测逻辑
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocessing.wear_dataset import CutterWearDataset

# 定义模型结构（需要与训练时保持一致）
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                 batch_first=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        out = self.fc(out)
        return out

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(torch.nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = torch.nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)
        out = self.dropout(transformer_out[:, -1, :])
        out = self.fc(out)
        return out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def load_model(model_path, model_type, input_size, output_size, device):
    """加载预训练模型"""
    try:
        # 使用默认配置创建模型
        default_config = {
            'hidden_size': 128,
            'num_layers': 2,
            'd_model': 128,
            'nhead': 8,
            'dropout': 0.2
        }
        
        # 创建模型
        if model_type == 'LSTM':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=default_config['hidden_size'],
                num_layers=default_config['num_layers'],
                output_size=output_size,
                dropout=default_config['dropout']
            )
        elif model_type == 'GRU':
            model = GRUModel(
                input_size=input_size,
                hidden_size=default_config['hidden_size'],
                num_layers=default_config['num_layers'],
                output_size=output_size,
                dropout=default_config['dropout']
            )
        elif model_type == 'RNN':
            model = RNNModel(
                input_size=input_size,
                hidden_size=default_config['hidden_size'],
                num_layers=default_config['num_layers'],
                output_size=output_size,
                dropout=default_config['dropout']
            )
        elif model_type == 'Transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=default_config['d_model'],
                nhead=default_config['nhead'],
                num_layers=default_config['num_layers'],
                output_size=output_size,
                dropout=default_config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, default_config
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

def predict_cutter_wear():
    """使用预训练模型预测刀具磨损"""
    
    parser = argparse.ArgumentParser(description="使用预训练模型预测刀具磨损")
    parser.add_argument('--input_csv', type=str, 
                       default=str(project_root / 'data/processed/wear_per_timestep.csv'), 
                       help='输入CSV文件路径')
    parser.add_argument('--output_csv', type=str, 
                       default=str(project_root / 'data/processed/predicted_wear_per_timestep.csv'), 
                       help='输出CSV文件路径')
    parser.add_argument('--model_path', type=str, 
                       default=str(project_root / 'checkpoints/wear_prediction/LSTM_dataset_1_400_checkpoint.pth'), 
                       help='预训练模型路径')
    parser.add_argument('--model_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'RNN', 'Transformer'],
                       help='模型类型')
    parser.add_argument('--start_ring', type=int, default=1, help='起始环号')
    parser.add_argument('--end_ring', type=int, default=400, help='结束环号')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--wear_method', type=str, default='radius', choices=['radius', 'volume'], 
                       help='磨损计算方法')
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not Path(args.input_csv).exists():
        print(f"错误: 输入文件不存在: {args.input_csv}")
        return
    
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 根据数据集确定输入输出维度
    # 创建一个小的样本数据集来获取维度信息
    temp_dataset = CutterWearDataset(
        csv_path=args.input_csv,
        sequence_length=100,
        start_ring=args.start_ring,
        end_ring=min(args.start_ring + 5, args.end_ring),  # 只用前几环来获取维度
        wear_method=args.wear_method,
        step_size=100,  # 推理模式：使用序列长度作为步长
        target_cutter=1  # 只预测1号刀具
    )
    
    if len(temp_dataset) == 0:
        print("错误: 无法创建临时数据集获取维度信息")
        return
        
    sample_x, sample_y = temp_dataset[0]
    input_size = sample_x.shape[1]  # 特征数量 (96)
    output_size = 1  # 只预测单把刀具
    
    print(f"检测到输入特征数: {input_size}")
    print(f"输出维度: {output_size} (预测目标刀具1号)")
    print(f"其他刀具将通过k值计算得到")
    
    # 读取原始数据
    print(f"读取数据从 {args.input_csv}...")
    original_df = pd.read_csv(args.input_csv)
    print(f"原始数据形状: {original_df.shape}")
    
    # 创建数据集（推理模式）
    SEQ_LENGTH = 100  # 必须与训练时保持一致
    STEP_SIZE = SEQ_LENGTH  # 推理模式：使用序列长度作为步长，为每个时间窗口生成预测
    
    print(f"创建数据集，序列长度: {SEQ_LENGTH}, 步长: {STEP_SIZE} (推理模式)")
    inference_dataset = CutterWearDataset(
        csv_path=args.input_csv,
        sequence_length=SEQ_LENGTH,
        start_ring=args.start_ring,
        end_ring=args.end_ring,
        wear_method=args.wear_method,
        step_size=STEP_SIZE,
        target_cutter=1  # 只预测1号刀具
    )
    
    if len(inference_dataset) == 0:
        print("错误: 数据集为空，无法进行预测")
        return
    
    print(f"数据集大小: {len(inference_dataset)}")
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载预训练模型
    print(f"加载模型从 {args.model_path}...")
    model, config = load_model(args.model_path, args.model_type, input_size, output_size, device)
    if model is None:
        return
    
    print(f"模型类型: {args.model_type}")
    print(f"输入特征数: {input_size}")
    print(f"输出维度: {output_size} (目标刀具1号)")
    
    # 进行推理
    print("开始预测...")
    all_predictions = []  # 只包含目标刀具的预测值
    all_targets = []  # 只包含目标刀具的真实值
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(inference_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # 预测（只预测最后一个时间步的目标刀具磨损）
            predictions = model(features)  # shape: (batch_size, 1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels[:, -1, :].cpu().numpy())  # 只取最后一个时间步的真实值
            
            if batch_idx % 100 == 0:
                print(f"已处理 {batch_idx + 1}/{len(inference_loader)} 批次")
    
    # 合并所有预测结果
    predictions_array = np.concatenate(all_predictions, axis=0)  # shape: (num_samples, 1)
    targets_array = np.concatenate(all_targets, axis=0)  # shape: (num_samples, 1)
    
    print(f"目标刀具预测结果形状: {predictions_array.shape}")
    print(f"目标刀具真实值形状: {targets_array.shape}")
    
    # 使用k值扩展到所有44把刀
    print("\n使用k值计算其他刀具的磨损...")
    all_cutters_predictions = inference_dataset.predict_other_cutters(
        torch.from_numpy(predictions_array).unsqueeze(1)  # shape: (num_samples, 1, 1)
    ).squeeze(1).numpy()  # shape: (num_samples, 44)
    
    print(f"所有刀具预测结果形状: {all_cutters_predictions.shape}")
    
    # 获取所有刀具的真实值（需要重新读取完整标签）
    print("获取所有刀具的真实值...")
    all_targets_full = []
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(inference_loader):
            # labels原本是(batch, seq, 1)，我们需要通过k值扩展得到完整的44把刀的真实值
            # 实际上，真实值也应该通过k值计算，因为数据集只返回目标刀具
            # 我们需要从原始CSV读取所有刀具的真实值
            pass
    
    # 从原始数据中提取所有刀具的真实值（用于评估）
    all_targets_full = []
    cutter_names = inference_dataset.label_columns  # 获取所有44个刀具的列名
    
    for i in range(len(predictions_array)):
        start_idx = i * STEP_SIZE
        end_idx = start_idx + SEQ_LENGTH
        
        if end_idx <= len(original_df):
            target_idx = end_idx - 1  # 预测的是序列的最后一个时间步
            target_values = original_df.loc[target_idx, cutter_names].values
            all_targets_full.append(target_values)
    
    all_targets_full = np.array(all_targets_full)  # shape: (num_samples, 44)
    print(f"所有刀具真实值形状: {all_targets_full.shape}")
    
    # 计算评估指标
    print("\n计算评估指标...")
    mse = mean_squared_error(all_targets_full, all_cutters_predictions)
    mae = mean_absolute_error(all_targets_full, all_cutters_predictions)
    rmse = np.sqrt(mse)
    
    # 计算每个刀具的R²分数
    r2_scores = []
    for i in range(all_targets_full.shape[1]):
        r2 = r2_score(all_targets_full[:, i], all_cutters_predictions[:, i])
        r2_scores.append(r2)
    
    mean_r2 = np.mean(r2_scores)
    
    print(f"MSE (所有刀具): {mse:.6f}")
    print(f"MAE (所有刀具): {mae:.6f}")
    print(f"RMSE (所有刀具): {rmse:.6f}")
    print(f"平均 R² (所有刀具): {mean_r2:.6f}")
    
    # 单独评估目标刀具 (1号刀)
    target_cutter_idx = 0  # 1号刀在列表中的索引
    target_mse = mean_squared_error(all_targets_full[:, target_cutter_idx], 
                                     all_cutters_predictions[:, target_cutter_idx])
    target_mae = mean_absolute_error(all_targets_full[:, target_cutter_idx], 
                                      all_cutters_predictions[:, target_cutter_idx])
    target_r2 = r2_scores[target_cutter_idx]
    
    print(f"\n目标刀具 (1号刀) 性能:")
    print(f"MSE: {target_mse:.6f}")
    print(f"MAE: {target_mae:.6f}")
    print(f"R²: {target_r2:.6f}")
    
    # 为原始数据添加预测列
    print("\n为原始数据添加预测磨损列...")
    
    # 获取刀具标签列名
    cutter_names = inference_dataset.label_columns
    
    # 初始化预测列
    for cutter_name in cutter_names:
        pred_col_name = f"predicted_{cutter_name}"
        original_df[pred_col_name] = np.nan
    
    # 填充预测值
    # 注意：使用序列长度作为步长，每个时间窗口不重叠
    prediction_counts = np.zeros(len(original_df))
    prediction_sums = np.zeros((len(original_df), len(cutter_names)))
    
    for i in range(len(all_cutters_predictions)):
        # 计算当前预测对应的原始数据位置（推理模式：非重叠窗口）
        start_idx = i * STEP_SIZE
        end_idx = start_idx + SEQ_LENGTH
        
        if end_idx <= len(original_df):
            # 将预测值添加到对应位置（为整个窗口的最后一个时间步）
            target_idx = end_idx - 1  # 预测的是序列的最后一个时间步
            prediction_sums[target_idx] += all_cutters_predictions[i]  # 使用k值计算后的所有刀具预测
            prediction_counts[target_idx] += 1
    
    # 计算预测值（推理模式：每个位置只有一个预测值）
    for i, cutter_name in enumerate(cutter_names):
        pred_col_name = f"predicted_{cutter_name}"
        mask = prediction_counts > 0
        original_df.loc[mask, pred_col_name] = prediction_sums[mask, i] / prediction_counts[mask]
    
    # 计算预测误差
    print("计算预测误差...")
    for cutter_name in cutter_names:
        pred_col_name = f"predicted_{cutter_name}"
        error_col_name = f"error_{cutter_name}"
        
        # 计算绝对误差
        original_df[error_col_name] = np.abs(original_df[cutter_name] - original_df[pred_col_name])
    
    # 保存结果
    print(f"保存预测结果到 {args.output_csv}...")
    try:
        original_df.to_csv(args.output_csv, index=False)
        print("成功保存预测结果！")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    # 生成预测结果摘要
    print("\n=== 预测结果摘要 ===")
    print(f"预测环号范围: {args.start_ring} - {args.end_ring}")
    print(f"磨损计算方法: {args.wear_method}")
    print(f"目标刀具: 1号刀 (直接预测)")
    print(f"其他刀具: 43把刀 (通过k值计算)")
    print(f"预测样本数量: {len(all_cutters_predictions)}")
    print(f"有效预测位置: {np.sum(prediction_counts > 0)}")
    
    # 显示每个刀具的R²分数
    print(f"\n各刀具预测性能 (R²分数):")
    for i, cutter_name in enumerate(cutter_names):
        if i < len(r2_scores):
            marker = " (直接预测)" if i == 0 else " (k值计算)"
            print(f"{cutter_name}: {r2_scores[i]:.4f}{marker}")
    
    print(f"\n预测完成！结果已保存到: {args.output_csv}")

def main():
    predict_cutter_wear()

if __name__ == '__main__':
    main()