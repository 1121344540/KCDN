
import argparse
import numpy as np
import logging

# 设置日志格式
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

# 数据集对应的评分文件名称
RATING_FILE_NAME = dict({'music': 'user_artists.dat', 'book': 'BX-Book-Ratings.csv', 'movie': 'ratings.csv'})
# 数据集的分隔符
SEP = dict({'music': '\t', 'book': ';', 'movie': ','})
# 评分阈值，高于该值的为正例
THRESHOLD = dict({'music': 0, 'book': 0, 'movie': 4})

# 读取物品索引到实体ID的映射文件
def read_item_index_to_entity_id_file(dataset):
    file = '../data/' + dataset + '/item_index2entity_id.txt'
    logging.info("reading item index to entity id file: %s", file)
    item_index_old2new = dict()
    entity_id2index = dict()
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return item_index_old2new, entity_id2index

# 将原始评分数据转换为正例和负例
def convert_rating(dataset, item_index_old2new, entity_id2index):
    file = '../data/' + dataset + '/' + RATING_FILE_NAME[dataset]
    logging.info("reading rating file: %s", file)
    
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    
    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[dataset])
        # 移除BX数据集的前缀和后缀引号
        if dataset == 'book':
            array = list(map(lambda x: x[1:-1], array))
        item_index_old = array[1]
        if item_index_old not in item_index_old2new.keys():  # 若物品不在最终的物品集中
            continue
        item_index = item_index_old2new[item_index_old]
        
        user_index_old = array[0]
        rating = float(array[2])
        if rating >= THRESHOLD[dataset]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)
    logging.info('converting rating file done.')
    return user_pos_ratings, user_neg_ratings

# 将原始的知识图谱数据转换为模型需要的格式
def convert_kg(dataset, entity_id2index):
    file = '../data/' + dataset + '/' + 'kg.txt'
    write_file = '../data/' + dataset + '/' + 'kg_final.txt'
    logging.info("converting kg file to: %s", write_file)
    
    entity_cnt = len(entity_id2index)
    relation_id2index = dict()
    relation_cnt = 0
    
    writer = open(write_file, 'w', encoding='utf-8')
    writer_idx = 0
    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))
        writer_idx += 1
    writer.close()
    
    logging.info("number of entities (containing items): %d", entity_cnt)
    logging.info("number of relations: %d", relation_cnt)
    logging.info("number of triples: %d", writer_idx)
    return entity_id2index, relation_id2index

if __name__ == '__main__':
    # 使用相同的随机种子以便与RippleNet, KGCN, KGNN-LS进行比较
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()

    item_index_old2new, entity_id2index = read_item_index_to_entity_id_file(args.dataset)
    convert_rating(args.dataset, item_index_old2new, entity_id2index)
    entity_id2index, relation_id2index = convert_kg(args.dataset, entity_id2index)

    logging.info("data %s preprocess: done.",args.dataset)
