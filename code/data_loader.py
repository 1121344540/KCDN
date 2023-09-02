import collections  # 引入collections库以便使用其高性能的数据结构。
import os  # 引入os库以便处理文件和目录路径。
import numpy as np  # 引入numpy库以便进行高性能的数学运算。
import logging  # 引入日志库以便记录日志信息。

# 设置日志的基本配置。
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    """加载数据的主函数。

    参数:
        args: 包含所有参数和配置的对象。

    返回:
        一个包含训练数据、评估数据、测试数据、实体数量、关系数量、用户三元组集合和物品三元组集合的元组。
    """
    logging.info("================== preparing data ===================")
    train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
    return train_data, eval_data, test_data, n_entity, n_relation, user_triple_sets, item_triple_sets

    """这一部分主要定义了一个名为load_data的函数，该函数负责加载数据，
    并进行相关的数据预处理操作。这个函数会从评分文件中加载训练、评估和测试数据，
    并加载知识图谱数据。然后，它会为用户和物品构建知识图谱三元组集合。"""



def load_rating(args):
    """加载评分数据的函数。

    参数:
        args: 包含所有参数和配置的对象。

    返回:
        一个包含训练数据、评估数据、测试数据、用户初始实体集合和物品初始实体集合的元组。
    """
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)
    """这个函数load_rating主要用于加载评分数据。
    它首先检查是否存在预先保存的numpy数组格式的评分文件，
    如果存在，则直接加载该文件。如果不存在，
    它会从文本文件中加载评分数据并将其保存为numpy数组格式以便于下次快速加载"""


def dataset_split(rating_np):
    """将评分数据集分为训练、评估和测试集。

    参数:
        rating_np: 包含评分数据的numpy数组。

    返回:
        一个包含训练数据、评估数据、测试数据、用户初始实体集合和物品初始实体集合的元组。
    """
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    user_init_entity_set, item_init_entity_set = collaboration_propagation(rating_np, train_indices)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]

    return rating_np[train_indices], rating_np[eval_indices], rating_np[
        test_indices], user_init_entity_set, item_init_entity_set
    """
    dataset_split函数的主要功能是将评分数据集随机分为训练、评估和测试集，其分割比例为6:2:2。
    此外，该函数还利用collaboration_propagation函数为每个用户和物品生成一个初始的实体集合。
    """


def collaboration_propagation(rating_np, train_indices):
    """基于协同过滤的传播函数，用于为每个用户和物品生成初始的实体集合。

    参数:
        rating_np: 包含评分数据的numpy数组。
        train_indices: 用于训练的数据的索引。

    返回:
        两个字典，分别为用户初始实体集合和物品初始实体集合。
    """
    # 该函数的具体实现需要根据`data_loader.py`文件的内容来确定。
    # 通常，此函数会遍历训练数据，并为每个用户和物品收集与其交互过的实体。
    # 最后，返回这些实体集合。

    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []
            item_history_user_dict[item].append(user)
        
    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict

    """
    collaboration_propagation函数的主要功能是为每个用户和物品生成初始的实体集合。
    这些实体集合是基于用户和物品在训练数据中的交互来确定的。
    通常，该函数会遍历训练数据，并为每个用户和物品收集与其交互过的实体。
    """


def load_kg(args):
    """加载知识图谱数据的函数。

    参数:
        args: 包含所有参数和配置的对象。

    返回:
        一个包含实体数量、关系数量和知识图谱的元组。
    """
    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg

    """
    load_kg函数主要用于加载知识图谱数据。
    它首先检查是否存在预先保存的numpy数组格式的知识图谱文件，如果存在，则直接加载该文件
    。如果不存在，它会从文本文件中加载知识图谱数据并将其保存为numpy数组格式以便于下次快速加载。
    接着，它会计算知识图谱中的实体和关系数量，并调用construct_kg函数来构建知识图谱。"""


def construct_kg(kg_np):
    """从numpy数组构建知识图谱的函数。

    参数:
        kg_np: 包含知识图谱数据的numpy数组。

    返回:
        一个知识图谱的字典，其中每个实体都映射到一个与其相关的实体-关系列表。
    """
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
        kg[tail].append((head, relation))
    return kg

    """
    construct_kg函数的主要功能是从提供的numpy数组中构建知识图谱。
    它创建了一个默认字典来存储与每个实体相关的其他实体和关系。
    """


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    """知识图谱传播函数，用于为对象（用户或物品）构建知识图谱三元组集合。

    参数:
        args: 包含所有参数和配置的对象。
        kg: 知识图谱。
        init_entity_set: 初始实体集合。
        set_size: 三元组集合的大小。
        is_user: 布尔值，表示对象是否为用户。

    返回:
        一个三元组集合的字典。
    """
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size]
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h, r, t = [], [], []
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])

            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace=(len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets


"""
kg_propagation函数的主要功能是为每个对象（用户或物品）生成与其相关的知识图谱三元组集合。
它首先确定需要考虑的层数（例如，与对象直接相关的实体或间接相关的实体），然后为每个层生成三元组。
"""
