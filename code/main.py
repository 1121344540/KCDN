# 导入所需库
import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

# 初始化命令行参数解析器
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument('-d', '--dataset', type=str, default='music', help='选择使用哪个数据集 (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=20, help='训练的周期数')
parser.add_argument('--batch_size', type=int, default=2048, help='每个批次的数据量')
parser.add_argument('--n_layer', type=int, default=2, help='层的深度')
parser.add_argument('--lr', type=float, default=0.002, help='学习率')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='L2正则化的权重')

parser.add_argument('--dim', type=int, default=160, help='实体和关系嵌入的维度')
parser.add_argument('--user_triple_set_size', type=int, default=16, help='用户三元组集合的大小')
parser.add_argument('--item_triple_set_size', type=int, default=100, help='项目三元组集合的大小')
parser.add_argument('--agg', type=str, default='concat', help='聚合器的类型 (sum, pool, concat)')

parser.add_argument('--use_cuda', type=bool, default=True, help='是否使用GPU')
parser.add_argument('--show_topk', type=bool, default=False, help='是否显示topk')
parser.add_argument('--random_flag', type=bool, default=False, help='是否使用随机种子')

# 解析命令行参数
args = parser.parse_args()

# 设置随机种子函数
def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  # 为numpy设置随机种子
    torch.manual_seed(torch_seed)            # 为torch设置随机种子
    torch.cuda.manual_seed(torch_seed)       # 为torch的cuda设置随机种子
    torch.cuda.manual_seed_all(torch_seed)   # 为所有torch的cuda设备设置随机种子

# 如果不使用随机种子，则设置特定的随机种子
if not args.random_flag:
    set_random_seed(304, 2019)

# 加载数据
data_info = load_data(args)

# 开始训练
train(args, data_info)
