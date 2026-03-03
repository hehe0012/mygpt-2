#!/usr/bin/env python3
"""
GPT-2 推理脚本 - 使用随机初始化参数
不加载预训练权重，直接使用模型架构进行推理
"""

import json
import os
import torch
import sys

# 导入模型和编码器
import sample
import encoder


def inference_with_random_params(
    model_name='124M',
    seed=42,
    nsamples=1,
    batch_size=1,
    length=40,
    temperature=0.8,
    top_k=40,
    top_p=1.0,
    models_dir='models',
    prompt="What is artificial intelligence?",
    device=None,
):
    """
    使用随机初始化参数进行推理
    :model_name: 模型大小 (124M, 355M 等)
    :seed: 随机种子
    :nsamples: 生成样本数
    :batch_size: 批大小
    :length: 生成的token数
    :temperature: 温度参数
    :top_k: top-k采样
    :top_p: nucleus采样
    :prompt: 输入提示词
    """
    
    # 设置随机种子
    torch.manual_seed(seed)
    
    # 检查设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print(f"模型大小: {model_name}")
    print(f"生成长度: {length}")
    print(f"温度: {temperature}, top_k: {top_k}")
    print(f"提示词: {prompt}\n")
    
    # 加载编码器
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    model_dir = os.path.join(models_dir, model_name)
    
    try:
        enc = encoder.get_encoder(model_name, models_dir)
    except Exception as e:
        print(f"编码器加载失败: {e}")
        return
    
    # 加载模型超参数
    hparams = sample.model.default_hparams()
    hparams_file = os.path.join(model_dir, 'hparams.json')
    if os.path.exists(hparams_file):
        with open(hparams_file) as f:
            hparams.override_from_dict(json.load(f))
        print(f"超参数配置: n_vocab={hparams.n_vocab}, n_embd={hparams.n_embd}, n_layer={hparams.n_layer}, n_head={hparams.n_head}")
    else:
        print(f"未找到hparams.json，使用默认超参数")
    
    # 创建模型实例（随机初始化，不加载权重）
    print("创建模型实例（使用随机参数）...")
    gpt_model = sample.model.GPT2Model(hparams)
    gpt_model.to(device)
    gpt_model.eval()
    
    print(f"模型参数量: {sum(p.numel() for p in gpt_model.parameters()):,}")
    
    # 编码提示词
    context_tokens = enc.encode(prompt)
    print(f"提示词tokens: {context_tokens[:20]}... (共{len(context_tokens)}个token)")
    
    # 执行推理
    print(f"\n开始生成...")
    try:
        with torch.no_grad():
            context = [context_tokens for _ in range(batch_size)]
            out = sample.sample_sequence(
                hparams=hparams,
                length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                model_instance=gpt_model,
                device=device,
            )[:, len(context_tokens):]
            
            out = out.detach().cpu().tolist()
            for i in range(batch_size):
                text = enc.decode(out[i])
                print("=" * 60)
                print(f"生成的文本 (样本 {i+1}):")
                print("=" * 60)
                print(prompt + text)
                print("=" * 60)
    except Exception as e:
        print(f"推理时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # 处理命令行参数
    prompt = "What is artificial intelligence?"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    
    inference_with_random_params(prompt=prompt)
