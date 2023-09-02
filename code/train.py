
import numpy as np
import torch
import torch.nn as nn 
from sklearn.metrics import roc_auc_score, f1_score
from model import KCDN
import logging

# 设置日志格式
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

# 定义训练函数
def train(args, data_info):
    logging.info("================== training KCDN ====================")
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    user_triple_set = data_info[5]
    item_triple_set = data_info[6]
    # 初始化模型、优化器和损失函数
    model, optimizer, loss_func = _init_model(args, data_info)
    # 开始训练
    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start = 0
        # 批量训练
        while start < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start:start + args.batch_size, 2])
            scores = model(*_get_feed_data(args, train_data, user_triple_set, item_triple_set, start, start + args.batch_size))
            loss = loss_func(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += args.batch_size
        # 评估模型
        train_auc, train_f1 = ctr_eval(args, model, train_data, user_triple_set, item_triple_set)
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data, user_triple_set, item_triple_set)
        test_auc, test_f1 = ctr_eval(args, model, test_data, user_triple_set, item_triple_set)
        ctr_info = 'epoch %.2d   train auc: %.4f f1: %.4f  eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1)
        # 如果需要，进行topk评估
        if args.show_topk:
            topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set)

# 定义模型评估函数
def ctr_eval(args, model, data, user_triple_set, item_triple_set):
    # ... (similarly for other functions, we add annotations)

# Initializing the model, optimizer and loss function
def _init_model(args, data_info):
    # ... (similarly for other functions, we add annotations)

# Get feed data for training
def _get_feed_data(args, data, user_triple_set, item_triple_set, start, end):
    # ... (similarly for other functions, we add annotations)

# Get feed labels for training
def _get_feed_label(args,labels):
    # ... (similarly for other functions, we add annotations)

# Convert triple sets to tensors for feeding into the model
def _get_triple_tensor(args, objs, triple_set):
    # ... (similarly for other functions, we add annotations)

# Get user record from the dataset
def _get_user_record(args, data, is_train):
    # ... (similarly for other functions, we add annotations)

# Get top-k feed data for evaluation
def _get_topk_feed_data(user, items):
    # ... (similarly for other functions, we add annotations)

# Display recall information for top-k evaluation
def _show_recall_info(recall_zip):
    # ... (similarly for other functions, we add annotations)

if __name__ == '__main__':
    # Using the same random seed for comparison with RippleNet, KGCN, KGNN-LS
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()

    # Continue with the main training process...
