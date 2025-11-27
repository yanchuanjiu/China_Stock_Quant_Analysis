#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易信号生成器

功能:
1. 基于模型预测生成买卖信号
2. 信号验证和过滤
3. 信号输出 (JSON/API)
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, config_path: str):
        """
        初始化信号生成器
        
        Args:
            config_path: 策略配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.dataset = None
        self._init_qlib()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _init_qlib(self):
        """初始化 Qlib"""
        provider_uri = Path.home() / ".qlib" / "qlib_data" / "cn_data"
        qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    def train(self, retrain: bool = False) -> None:
        """
        训练模型
        
        Args:
            retrain: 是否强制重新训练
        """
        data_config = self.config['data']
        model_config = self.config['model']
        factor_config = self.config['factors']
        
        # 数据处理器配置
        handler_config = {
            "class": factor_config['handler'],
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": data_config['train_start'],
                "end_time": data_config['test_end'],
                "fit_start_time": data_config['train_start'],
                "fit_end_time": data_config['train_end'],
                "instruments": data_config['instruments'],
            },
        }
        
        # 数据集配置
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (data_config['train_start'], data_config['valid_start']),
                    "valid": (data_config['valid_start'], data_config['valid_end']),
                    "test": (data_config['test_start'], data_config['test_end']),
                },
            },
        }
        
        # 初始化模型和数据集
        self.model = init_instance_by_config({
            "class": model_config['class'],
            "module_path": model_config['module_path'],
            "kwargs": model_config['params'],
        })
        self.dataset = init_instance_by_config(dataset_config)
        
        # 训练
        print(f"🚀 训练模型: {self.config['strategy']['name']}")
        self.model.fit(self.dataset)
        print("✅ 训练完成")
    
    def generate_signals(self, date: str = None) -> Dict:
        """
        生成交易信号
        
        Args:
            date: 信号日期，默认为最新日期
            
        Returns:
            信号字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")
        
        # 预测
        pred = self.model.predict(self.dataset)
        
        # 获取最新日期的预测
        if date is None:
            date = pred.index.get_level_values('datetime').max()
        
        # 筛选当日预测
        daily_pred = pred.xs(date, level='datetime')
        
        # 排序并选择 Top K
        topk = self.config['backtest']['strategy']['topk']
        top_stocks = daily_pred.nlargest(topk, 'score')
        
        # 计算权重
        scores = top_stocks['score'].values
        weights = np.exp(scores) / np.exp(scores).sum()  # Softmax 权重
        
        # 获取当前价格
        stocks = top_stocks.index.tolist()
        prices = D.features(stocks, ['$close'], start_time=date, end_time=date, freq='day')
        
        # 生成信号
        account = self.config['backtest']['account']
        signals = []
        
        for i, (stock, row) in enumerate(top_stocks.iterrows()):
            price = prices.loc[(date, stock), '$close'] if (date, stock) in prices.index else 0
            target_value = account * weights[i]
            target_amount = int(target_value / price / 100) * 100 if price > 0 else 0
            
            signals.append({
                "stock": stock,
                "action": "BUY",
                "weight": float(weights[i]),
                "score": float(row['score']),
                "price_ref": float(price),
                "target_amount": target_amount,
            })
        
        # 构建输出
        result = {
            "date": date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10],
            "strategy": self.config['strategy']['name'],
            "version": self.config['strategy']['version'],
            "generated_at": datetime.now().isoformat(),
            "signals": signals,
            "portfolio_summary": {
                "total_buy_value": sum(s['price_ref'] * s['target_amount'] for s in signals),
                "stock_count": len(signals),
                "top_score": float(top_stocks['score'].max()),
                "avg_score": float(top_stocks['score'].mean()),
            }
        }
        
        return result
    
    def save_signals(self, signals: Dict, output_dir: str = None) -> str:
        """
        保存信号到文件
        
        Args:
            signals: 信号字典
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        if output_dir is None:
            output_dir = self.config['signal']['output_dir']
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f"{signals['date']}_{signals['strategy']}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(signals, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 信号已保存: {filepath}")
        return str(filepath)
    
    def validate_signals(self, signals: Dict) -> Tuple[bool, List[str]]:
        """
        验证信号有效性
        
        Args:
            signals: 信号字典
            
        Returns:
            (是否有效, 警告列表)
        """
        warnings = []
        
        # 检查信号数量
        if len(signals['signals']) == 0:
            warnings.append("⚠️ 无买入信号")
            return False, warnings
        
        # 检查权重总和
        total_weight = sum(s['weight'] for s in signals['signals'])
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(f"⚠️ 权重总和 {total_weight:.4f} != 1.0")
        
        # 检查单股权重
        risk_config = self.config.get('risk', {})
        position_limit = risk_config.get('position_limit', 0.2)
        
        for s in signals['signals']:
            if s['weight'] > position_limit:
                warnings.append(f"⚠️ {s['stock']} 权重 {s['weight']:.2%} 超过上限 {position_limit:.0%}")
        
        # 检查价格有效性
        for s in signals['signals']:
            if s['price_ref'] <= 0:
                warnings.append(f"⚠️ {s['stock']} 价格无效")
        
        return len(warnings) == 0, warnings


def main():
    """测试信号生成"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "strategies" / "volume_factor_lgb.yaml"
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    generator = SignalGenerator(str(config_path))
    
    # 训练模型
    generator.train()
    
    # 生成信号
    signals = generator.generate_signals()
    
    # 验证信号
    valid, warnings = generator.validate_signals(signals)
    if warnings:
        for w in warnings:
            print(w)
    
    # 保存信号
    output_path = generator.save_signals(signals)
    
    # 打印信号摘要
    print("\n" + "=" * 60)
    print(f"📊 信号摘要 - {signals['date']}")
    print("=" * 60)
    print(f"策略: {signals['strategy']}")
    print(f"股票数: {signals['portfolio_summary']['stock_count']}")
    print(f"最高分: {signals['portfolio_summary']['top_score']:.4f}")
    print(f"平均分: {signals['portfolio_summary']['avg_score']:.4f}")
    print(f"\n买入信号:")
    for s in signals['signals']:
        print(f"  {s['stock']}: 权重={s['weight']:.2%}, 分数={s['score']:.4f}, 目标={s['target_amount']}股 @{s['price_ref']:.2f}")


if __name__ == "__main__":
    main()

