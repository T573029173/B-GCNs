"""
# -*- coding:utf-8 -*-
# Version:Python3.8.3
# @Time:2022/7/20 7:40
# @Author:Lzy
# @File:B_GCNs_util.py
# @Software:PyCharm
# Code Description: 
"""
import xlrd
import numpy as np
import scipy.sparse as sp
import torch

# ------------------- 数据预处理（主要包括构建构建采样点的图结构；获取待插值点坐标、图结构）


# 读取数据
def read_data(path):
    print('第一步：数据正在读取...')
    ''' 设置x、y、z坐标间距 '''
    x_length = 10
    y_length = 10
    z_length = 1
    ''' 设置搜索距离 '''
    search_x = 180
    search_y = 180
    search_z = 7
    workbook = xlrd.open_workbook(path)
    sheet_name = workbook.sheet_names()[0]
    host_sheets = workbook.sheet_by_name(sheet_name)
    rows = host_sheets.nrows
    original_data = []   # 原始数据
    for o in range(rows):
        original_data += [host_sheets.row_values(o)]
    print('---数据读取完毕！')
    data_id = []  # 获取已知点的id
    data_type = []  # 获取已知点类型
    data_point = []  # 获取已知点坐标
    for p in range(1, len(original_data)):
        d_p = [original_data[p][1], original_data[p][2], original_data[p][3]]
        data_id.append(original_data[p][0])
        data_point.append(d_p)
        data_type.append(original_data[p][5])
    # print(original_data)
    # 构建已知点位邻接关系——邻接矩阵  用于构建训练集和验证集
    known_neighbor_data, known_neighbor_id_data = \
        known_neighbor_distribution(original_data, search_x, search_y, search_z)
    # 创建渔网，获取待插值点的坐标
    point_x, point_y, point_z = create_fish_net(original_data, x_length, y_length, z_length)
    # 初步判定待插值点位与已知点的邻接关系
    unknown_p_0, unknown_p_2 = \
        unknown_neighbor(original_data, point_x, point_y, point_z, search_x, search_y, search_z)
    # 构建待插值点位与已知点的邻接矩阵——未知点邻接矩阵  用于构建预测集
    unknown_k_neighbor, known_neighbor_id = \
        unknown_neighbor_distribution(original_data, unknown_p_2,
                                      search_x, search_y, search_z)
    return data_point, data_id, data_type, known_neighbor_data, known_neighbor_id_data,\
        unknown_k_neighbor, known_neighbor_id, unknown_p_2, unknown_p_0


# 构建已知点位邻接关系——已知点邻接矩阵  用于构建训练集和验证集
def known_neighbor_distribution(data, s_x, s_y, s_z):
    """
    构建已知点位邻接关系——邻接矩阵
    :param data: 原始数据
    :param s_x: x轴搜索距离
    :param s_y: y轴搜索距离
    :param s_z: z轴搜索距离
    :return: 已知点位在搜索距离之内的连接关系--已知点的邻接矩阵，1代表相邻，0代表不相邻
    """
    print('第二步：已知点位的邻接关系正在构建中...')
    data_id = []  # id数据
    data_x = []  # x轴坐标数据
    data_y = []  # y轴坐标数据
    data_z = []  # z轴坐标数据
    known_neighbor = []  # 已知点的邻居,存储的是0-1值，1为相邻，0为不相邻。
    known_neighbor_id = []  # 已知点邻居的ID
    known_neighbor_id_0 = []  # 邻居为0的已知点ID
    # 分别提取出各轴坐标数据
    for p in range(1, len(data)):
        data_id.append(data[p][0])
        data_x.append(data[p][1])
        data_y.append(data[p][2])
        data_z.append(data[p][3])
    # print(data_x)
    # 获取已知点的邻居id
    for k in range(len(data_x)):
        k_n = []  # 存储已知点邻居的是否相邻的判断类型（0-1）
        kk = []
        k_n_id = []  # 存储已知点邻居的ID
        for r in range(len(data_x)):
            if k != r and (abs(data_x[k] - data_x[r]) <= s_x) \
                    and (abs(data_y[k] - data_y[r]) <= s_y) \
                    and (abs(data_z[k] - data_z[r]) <= s_z):
                kk.append(1)
                k_n.append("1")
                k_n_id.append(data_id[r])
            else:
                kk.append(0)
                k_n.append("0")
        if sum(kk) > 0:
            known_neighbor.append(k_n)
            known_neighbor_id.append(k_n_id)
        else:
            known_neighbor_id_0.append(k + 1)
            print("ID号为:{}的邻居为0".format(k + 1))
    print("共有{}个已知点位的邻居为0".format(len(known_neighbor_id_0)))
    print('---已知点位的邻接关系构建完毕！')
    return known_neighbor, known_neighbor_id


# 创建渔网，获取待插值的点位坐标
def create_fish_net(data, d_x, d_y, d_z):
    """
    本函数用于创建渔网，获取待插值的点位坐标
    :param data: 原始数据
    :param d_x: x轴坐标间隔
    :param d_y: y轴坐标间隔
    :param d_z: z轴坐标间隔
    :return: 返回渔网的点位坐标
    """
    print('第三步：渔网正在创建中...')
    # 获取各轴的最大值、最小值坐标以及两者之差
    data_x = []  # x轴坐标数据
    data_y = []  # y轴坐标数据
    data_z = []  # z轴坐标数据
    for p in range(1, len(data)):
        data_x.append(data[p][1])
        data_y.append(data[p][2])
        data_z.append(data[p][3])
    max_min_x = max(data_x) - min(data_x)  # X轴的长度
    max_min_y = max(data_y) - min(data_y)  # Y轴的长度
    max_min_z = max(data_z) - min(data_z)  # Z轴的长度
    print('x轴的最大值：', max(data_x), '最小值为：', min(data_x), '两者的差为：', max_min_x,
          '\ny轴的最大值：', max(data_y), '最小值为：', min(data_y), '两者的差为：', max_min_y,
          '\nz轴的最大值：', max(data_z), '最小值为：', min(data_z), '两者的差为：', max_min_z)
    # 创建渔网坐标
    # x轴坐标
    # np.linspace 定间隔取点
    fish_net_x = np.linspace(0, int(max_min_x + d_x), num=int(max_min_x / d_x), endpoint=False)
    fish_net_x = [x + min(data_x) for x in fish_net_x]
    print("X轴坐标个数", len(fish_net_x))
    # y轴坐标
    fish_net_y = np.linspace(0, int(max_min_y + d_y), num=int(max_min_y / d_y), endpoint=False)
    fish_net_y = [y + min(data_y) for y in fish_net_y]
    print("Y轴坐标个数", len(fish_net_y))
    # z轴坐标
    fish_net_z = np.linspace(0, int(max(data_z) + d_z), num=int(max(data_z) / d_z), endpoint=False, dtype=int)
    fish_net_z = [z + min(data_z) for z in fish_net_z]
    print("Z轴坐标个数", len(fish_net_z))
    print('---渔网创建完毕，坐标已获取！')
    return fish_net_x, fish_net_y, fish_net_z


# 初步判定待插值点位与已知点的邻接关系
def unknown_neighbor(data, p_x, p_y, p_z, s_x, s_y, s_z):
    """
    构建待插值点位的已知点的邻接关系——未知点邻接矩阵
    :param data: 原始数据
    :param p_x: 待插值点位的x坐标
    :param p_y: 待插值点位的y坐标
    :param p_z: 待插值点位的z坐标
    :param s_x: x轴搜索距离
    :param s_y: y轴搜索距离
    :param s_z: z轴搜索距离
    :return: 待插值点位和已知点位的邻接关系——未知点邻接矩阵，1代表相邻，0代表不相邻
    """
    print('第四步：待插值点位的邻接关系正在构建中...')
    data_id = []  # 已知点位的id数据
    data_x = []  # x轴坐标数据
    data_y = []  # y轴坐标数据
    data_z = []  # z轴坐标数据
    unknown_point_0 = []  # 初次判定待插值点位的污染类型，原理：当待插值点位与所有已知点位在搜索范围内没有相邻关系，即判定该点位为无污染类型. 存储内容的为坐标点位
    unknown_point_2 = []  # 存储与已知点位具有相邻关系的待插值点位的坐标
    # 获取已知点位的各轴坐标及点位id
    for p in range(1, len(data)):
        data_id.append(data[p][0])
        data_x.append(data[p][1])
        data_y.append(data[p][2])
        data_z.append(data[p][3])
    # 获取未知点位的邻居（已知点位）
    for px in range(len(p_x)):
        for py in range(len(p_y)):
            for pz in range(len(p_z)):
                p_neigh_relations = []  # 存储待插值点与已知点的相邻关系(0-1)
                p_neigh_relations_know_id = []  # 存储待插值点与已知点是相邻关系的id（已知点的id）
                for d in range(len(data_x)):
                    if abs(p_x[px] - data_x[d]) <= s_x and \
                            abs(p_y[py] - data_y[d]) <= s_y and \
                            abs(p_z[pz] - data_z[d]) <= s_z:
                        p_neigh_relations.append(1)
                        p_neigh_relations_know_id.append(data_id[d])
                    else:
                        p_neigh_relations.append(0)
                type_un = np.sum(p_neigh_relations)
                un_point = [p_x[px], p_y[py], p_z[pz]]
                if type_un == 0:
                    unknown_point_0.append(un_point)  # 存储与已知点位的在搜索距离内无邻居关系的待插值点位的坐标
                else:
                    unknown_point_2.append(un_point)
    print('---待插值点位的邻接关系构建完毕！')
    return unknown_point_0, unknown_point_2


# 构建待插值点位与已知点的邻接矩阵——未知点邻接矩阵  用于构建预测集
def unknown_neighbor_distribution(data, unknown_p_2, s_x, s_y, s_z):
    print('第六步：待插值点位的邻接关系正在构建中...')
    unknown_k_neighbor = []  # 待插值点位的邻居，存储的是0-1值，1代表相邻，0代表不相邻，顺序与known_neighbor_id一致
    known_neighbor_id = []  # 待插值点位的邻居，存储的是已知点位的id，顺序与unknown_neighbor一致
    # 构建待插值点位的邻接矩阵
    ss_multiple = int(len(unknown_p_2) / (len(data) - 1))
    unknown_point_list = []  # 将待插值点位分割，构建邻接矩阵，分割数量为已知点数量，不足的取最后的数值
    for ss in range(ss_multiple + 1):
        unknown_p = unknown_p_2[ss * (len(data) - 1):(ss + 1) * (len(data) - 1)]
        if len(unknown_p) != len(data) - 1:
            l_extend = unknown_p_2[len(unknown_p_2) - (len(data) - 1):ss * (len(data) - 1)]
            unknown_p.extend(l_extend)
        unknown_point_list.append(unknown_p)

    for c in range(len(unknown_point_list)):
        point_relation = []
        point_relation_id = []
        for s in range(len(unknown_point_list[c])):
            p_neigh_r = []  # 存储待插值点与已知点的相邻关系(0-1)
            p_neigh_r_k_id = []  # 存储待插值点与已知点是相邻关系的id（已知点的id）
            for d in range(1, len(data)):
                if abs(unknown_point_list[c][s][0] - data[d][1]) <= s_x and \
                        abs(unknown_point_list[c][s][1] - data[d][2]) <= s_y and \
                        abs(unknown_point_list[c][s][2] - data[d][3]) <= s_z:
                    p_neigh_r.append("1")
                    p_neigh_r_k_id.append(int(data[d][0]))
                else:
                    p_neigh_r.append("0")
            point_relation.append(p_neigh_r)
            point_relation_id.append(p_neigh_r_k_id)
        unknown_k_neighbor.append(point_relation)
        known_neighbor_id.append(point_relation_id)
    print('---待插值点位的邻接关系构建完毕！')
    return unknown_k_neighbor, known_neighbor_id

# ------------------- 数据预处理完毕！
# ------------------- 将预处理的数据导入预测模型中。。。


# 预测精度计算
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    # sd = correct / len(labels)
    return correct / len(labels)


# 邻接矩阵数据归一化
def normalize_adj(adjacency):
    #  sp.eye()对角有1的稀疏矩阵
    #  eye创建单位矩阵，第一个参数为行数，第二个为列数
    adjacency += sp.eye(adjacency.shape[0])
    #  将稀疏矩阵中的data数据转为数组
    degree = np.array(adjacency.sum(1))
    #  sp.diags()  稀疏矩阵对角化  提取矩阵中对角值
    # 构建对角元素为ad的对角矩阵
    #  np.power() 用于数组元素求n次方
    #  np.flatten().该函数返回一个折叠成一维的数组
    # 用对角矩阵与原始矩阵的点积起到标准化的作用
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    # .dot() 为数组相乘
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


# 特征数据归一化
def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# 特征独热码处理
def encode_one_hot(labels):
    """
    在很多的多分类问题中，特征的标签通常都是不连续的内容（如本文中特征是离散的字符串类型），为了便于后续的计算、处理，
    需要将所有的标签进行提取，并将标签映射到一个独热码向量中。
    :param labels:
    :return:
    """
    # 将所有的标签整合成一个不重复的列表
    # classes = set(labels)  # set()函数创建一个无序不重复元素集
    classes = ['safety', 'pollution']
    '''
    enumerate()函数生成序列，带有索引i和值c。
    这一句将string类型的label变成int类型的label，建立映射关系
    np.identity(len(classes)) 为创建一个classes的单位矩阵
    创建一个字典，索引为label，值为独热码向量(就是之前生成的矩阵中的某一行)
    '''
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}

    # 为所有标签生成相应的独热码
    # map() 会根据提供的函数对指定序列做映射
    # 这一句将string类型的label替换为int类型的label
    labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_one_hot


# 加载训练、测试及验证数据
def load_data(d_type, k_n, k_n_id):
    # 存储为csr型的稀疏矩阵
    features = sp.csr_matrix(k_n, dtype=np.float32)
    #  获取标签数据，共2个标签
    #  标签 根据标签数据对每一行进行0-1值转换，标签对应的为1，不对应的为0
    labels = encode_one_hot(d_type)
    # build graph
    edges = []
    for h in range(len(k_n_id)):
        for f in range(len(k_n_id[h])):
            ff = [h, int(k_n_id[h][f] - 1)]
            edges.append(ff)
    edges = np.array(edges)
    #  coo_matrix  coo型稀疏矩阵
    #  根据coo矩阵性质，这段的作用就是，网络有多少条边，邻接矩阵就有多少个1
    #  所以先创建一个长度为edges_num的全1数组，每个1的填充位置就是一条边中两个端点的编号
    #  即edges[:,0],edges[:,1],矩阵的形状为（node_size，nose_size）
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #  对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = normalize_adj(adj)
    #  分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入
    rand_indices = np.random.permutation(len(d_type))
    # DBA
    idx_train = rand_indices[:55]
    idx_val = rand_indices[55:73]
    idx_test = rand_indices[73:len(d_type)]

    # BaP
    # idx_train = rand_indices[:48]
    # idx_val = rand_indices[48:64]
    # idx_test = rand_indices[64:len(d_type)]

    #  tensor为pytorch常用的数据结构
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)