#!/usr/bin/env python3
"""
刀具磨损预测模型训练脚本
使用LSTM, GRU, RNN, Transformer等经典时序预测模型
网络直接预测所有刀具的磨损值（使用真实值修正后的数据）
"""

# 设置matplotlib后端为非交互式，避免阻塞
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入自定义数据集
from wear_dataset import CutterWearDataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class LSTMModel(nn.Module):
    """LSTM模型 - 增强版"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # 增加全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class GRUModel(nn.Module):
    """GRU模型 - 增强版"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # 增加全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.dropout(gru_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class RNNModel(nn.Module):
    """RNN模型 - 增强版"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.dropout = nn.Dropout(dropout)
        # 增加全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.dropout(rnn_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)
        # 取最后一个时间步的输出
        out = self.dropout(transformer_out[:, -1, :])
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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

class CutterWearPredictor:
    """刀具磨损预测器"""
    
    def __init__(self, model_type='LSTM', config=None, models_dir=None, results_dir=None):
        self.model_type = model_type
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': []}
        self.models_dir = Path(models_dir) if models_dir else Path('models')
        self.results_dir = Path(results_dir) if results_dir else Path('results')
        
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'hidden_size': 256,
            'num_layers': 3,
            'd_model': 256,
            'nhead': 8,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 150,
            'patience': 15,
        }
    
    def create_model(self, input_size, output_size):
        """创建模型"""
        if self.model_type == 'LSTM':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            )
        elif self.model_type == 'GRU':
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            )
        elif self.model_type == 'RNN':
            model = RNNModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            )
        elif self.model_type == 'Transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                output_size=output_size,
                dropout=self.config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
        return model.to(device)
    
    def prepare_data(self, dataset, train_ratio=0.8):
        """准备训练和验证数据"""
        # 数据集划分
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, input_size, output_size):
        """训练模型 - output_size=44 表示预测所有刀具的磨损"""
        # 创建模型
        self.model = self.create_model(input_size, output_size)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练 {self.model_type} 模型...")
        print(f"输入维度: {input_size}, 输出维度: {output_size} (所有刀具磨损)")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # 预测最后一个时间步所有刀具的磨损值
                # batch_y shape: (batch_size, seq_len, num_cutters) -> (batch_size, num_cutters)
                batch_y = batch_y[:, -1, :]  # shape: (batch_size, output_size)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    batch_y = batch_y[:, -1, :]
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 记录损失
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f'best_{self.model_type.lower()}_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= self.config['patience']:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= self.config['patience']:
                print(f"早停于第 {epoch+1} 轮")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'best_{self.model_type.lower()}_model.pth'))
        print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return best_val_loss
    
    def evaluate_model(self, test_loader, dataset=None):
        """评估模型 - 评估所有刀具的预测准确性"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y[:, -1, :]  # 所有刀具的真实磨损
                
                outputs = self.model(batch_x)  # 所有刀具的预测磨损
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # 计算所有刀具的整体评估指标
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(targets.flatten(), predictions.flatten())
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'num_cutters': predictions.shape[1] if len(predictions.shape) > 1 else 1,
        }
        
        # 添加预测和真实值用于后续分析
        results['predictions'] = predictions
        results['targets'] = targets
        
        return results
    
    def train_and_evaluate(self, dataset, dataset_name, input_size, output_size, train_ratio=0.8):
        """完整的训练和评估流程"""
        print(f"\n训练 {self.model_type} 模型 (数据集: {dataset_name})...")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(dataset, train_ratio)
        
        # 训练模型
        best_val_loss = self.train_model(train_loader, val_loader, input_size, output_size)
        
        # 评估模型
        eval_results = self.evaluate_model(val_loader)
        
        # 保存模型
        model_save_path = self.models_dir / f"{self.model_type}_{dataset_name}_checkpoint.pth"
        torch.save(self.model.state_dict(), model_save_path)
        
        # 保存配置
        config_save_path = self.models_dir / f"{self.model_type}_{dataset_name}_info.json"
        with open(config_save_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'dataset_name': dataset_name,
                'config': self.config,
                'best_val_loss': best_val_loss,
                'mse': eval_results['mse'],
                'mae': eval_results['mae'],
                'rmse': eval_results['rmse'],
                'r2': eval_results['r2'],
                'input_size': input_size,
                'output_size': output_size
            }, f, indent=4)
        
        # 绘制训练历史
        self.plot_training_history(dataset_name=dataset_name)
        
        # 打印结果
        print(f"  MSE: {eval_results['mse']:.6f}")
        print(f"  MAE: {eval_results['mae']:.6f}")
        print(f"  RMSE: {eval_results['rmse']:.6f}")
        print(f"  R²: {eval_results['r2']:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'eval_results': eval_results,
            'config': self.config
        }
    
    def plot_training_history(self, dataset_name=None):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        
        title = f'{self.model_type} Training History'
        if dataset_name:
            title += f' - {dataset_name}'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        filename = f'{self.model_type}_{dataset_name}_training_history.png' if dataset_name else f'{self.model_type}_training_history.png'
        save_path = self.results_dir / filename
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存到: {save_path}")

def create_datasets():
    """创建三个数据集 - 预测所有刀具的磨损（使用修正后的数据）"""
    csv_path = '/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/data/processed/wear_per_timestep_corrected.csv'
    
    datasets = {}
    dataset_configs = [
        {'name': 'dataset_1_400', 'start_ring': 1, 'end_ring': 400},
        {'name': 'dataset_300_700', 'start_ring': 300, 'end_ring': 700},
        {'name': 'dataset_530_930', 'start_ring': 530, 'end_ring': 930}
    ]
    
    for config in dataset_configs:
        print(f"创建数据集 {config['name']}: 环号 {config['start_ring']}-{config['end_ring']}")
        dataset = CutterWearDataset(
            csv_path=csv_path,
            sequence_length=100,
            start_ring=config['start_ring'],
            end_ring=config['end_ring'],
            step_size=10,
            normalize=True
        )
        
        if len(dataset) > 0:
            datasets[config['name']] = dataset
            print(f"  数据集大小: {len(dataset)} 样本")
            print(f"  特征维度: {len(dataset.feature_columns)}")
            print(f"  输出维度: {dataset.num_cutters} (所有刀具)")
        else:
            print(f"  警告: 数据集 {config['name']} 为空")
    
    return datasets

def main():
    """主函数"""
    print("="*50)
    print("刀具磨损预测模型训练")
    print("="*50)
    
    # 创建数据集
    datasets = create_datasets()
    
    if not datasets:
        print("错误: 所有数据集都为空")
        return
    
    # 模型列表
    models = ['LSTM', 'GRU', 'RNN', 'Transformer']
    
    # 结果存储
    results = {}
    
    # 创建保存目录
    models_dir = Path('/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/models')
    results_dir = Path('/media/young/4A8C40028C3FE75B/刀盘数据/disc_wear/results')
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存路径设置:")
    print(f"  模型文件: {models_dir}")
    print(f"  训练图片: {results_dir}")
    
    # 对每个数据集和模型进行训练
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*30}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*30}")
        
        # 获取数据维度
        sample_x, sample_y = dataset[0]
        input_size = sample_x.shape[1]  # 特征数量 (96)
        output_size = sample_y.shape[1]  # 输出维度 (44, 所有刀具)
        
        print(f"输入特征数: {input_size}")
        print(f"输出维度: {output_size} (所有刀具)")
        
        results[dataset_name] = {}
        
        for model_type in models:
            print(f"\n训练 {model_type} 模型...")
            
            # 创建预测器
            predictor = CutterWearPredictor(model_type=model_type)
            
            # 准备数据
            train_loader, val_loader = predictor.prepare_data(dataset, train_ratio=0.8)
            
            # 训练模型
            try:
                best_val_loss = predictor.train_model(
                    train_loader, val_loader, input_size, output_size
                )
                
                # 评估模型
                eval_results = predictor.evaluate_model(val_loader)
                
                # 保存结果
                results[dataset_name][model_type] = {
                    'best_val_loss': best_val_loss,
                    'eval_results': eval_results,
                    'config': predictor.config
                }
                
                # 保存模型到 models/ 目录
                model_save_path = models_dir / f"{model_type}_{dataset_name}_checkpoint.pth"
                torch.save(predictor.model.state_dict(), model_save_path)
                
                # 保存配置到 models/ 目录
                config_save_path = models_dir / f"{model_type}_{dataset_name}_info.json"
                with open(config_save_path, 'w') as f:
                    json.dump({
                        'model_type': model_type,
                        'dataset_name': dataset_name,
                        'config': predictor.config,
                        'best_val_loss': best_val_loss,
                        'mse': eval_results['mse'],
                        'mae': eval_results['mae'],
                        'rmse': eval_results['rmse'],
                        'r2': eval_results['r2'],
                        'input_size': input_size,
                        'output_size': output_size
                    }, f, indent=4)
                
                # 绘制训练历史（保存图片到 results/ 目录，不显示）
                predictor.plot_training_history(save_dir=results_dir, dataset_name=dataset_name)
                
                print(f"  MSE: {eval_results['mse']:.6f}")
                print(f"  MAE: {eval_results['mae']:.6f}")
                print(f"  RMSE: {eval_results['rmse']:.6f}")
                print(f"  R²: {eval_results['r2']:.6f}")
                
            except Exception as e:
                print(f"  训练 {model_type} 时出错: {e}")
                results[dataset_name][model_type] = {'error': str(e)}
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("训练结果汇总")
    print("="*50)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n数据集: {dataset_name}")
        print("-" * 40)
        for model_type, result in dataset_results.items():
            if 'error' in result:
                print(f"{model_type:12}: 训练失败 - {result['error']}")
            else:
                eval_res = result['eval_results']
                print(f"{model_type:12}: RMSE={eval_res['rmse']:.4f}, R²={eval_res['r2']:.4f}")
    
    print(f"\n保存位置:")
    print(f"  模型文件(.pth, .json): {models_dir}")
    print(f"  训练图片(.png): {results_dir}")
    print("\n训练完成！")

if __name__ == "__main__":
    main()