"""
# -*- coding:utf-8 -*-
# Version:Python3.8.3
# @Time:2022/7/20 7:41
# @Author:Lzy
# @File:B_GCNs_predict.py
# @Software:PyCharm
# Code Description: 
"""
import time
import torch
import argparse
import xlsxwriter
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from B_GCNs_model import B_Gcns_Net
from B_GCNs_util import load_data, accuracy, read_data, normalize, sparse_mx_to_torch_sparse_tensor

# 训练超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=20,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
b_name = "DBA"
w_decay = 4  # 权重幂数
cell_size_l = 180  # 格网大小-水平
cell_size_v = 7  # 格网大小-垂直
t_value = 1  # 阈值倍数

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_path = r'data/DBA_data.xlsx'
# data_path = r'data/Bap_data.xlsx'


# 加载原始数据
original_data, data_id, data_type, known_neighbor_data, known_neighbor_id_data, unknown_k_neighbor, known_neighbor_id, \
  unknown_point_list, unknown_point_0 = read_data(data_path)


# 加载训练、测试及验证数据
adjacency, features, labels, idx_train, idx_val, idx_test = \
    load_data(data_type, known_neighbor_data, known_neighbor_id_data)

# Model and optimizer
model = B_Gcns_Net(input_dim=features.shape[1],
               hidden_dim=args.hidden,
               output_dim=labels.max().item() + 1,
               dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adjacency = adjacency.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# 计算训练和验证数据的先验概率（考虑土壤因素）
def soil_prior_probability():
    label_probability = []  # 用于存储标签的先验概率
    soil_label_1 = 2 / 3  # 经地理探测器所计算，标签为1的先验概率
    soil_label_0 = 1 / 3  # 经地理探测器所计算，标签为0的先验概率

    # 重新整理标签的先验概率
    for lp in range(labels.shape[0]):
        l_p = []
        if labels[lp].item() == 1:
            l_p.append(soil_label_1)
            l_p.append(soil_label_0)
        else:
            l_p.append(soil_label_0)
            l_p.append(soil_label_1)
        label_probability.append(l_p)
    return label_probability


# 训练主体函数
def train(prior_prob_tensor):
    train_loss_history = []  # 用于存储每次训练误差值
    train_acc_history = []  # 用于存储每次训练精度值
    val_loss_history = []  # 用于存储每次验证误差值
    val_acc_history = []  # 用于存储每次验证精度值
    test_loss_history = []  # 用于存储每次测试误差值
    test_acc_history = []  # 用于存储每次测试精度值
    val_label_history = []  # 用于存储每次验证结果（标签）
    train_label_history = []  # 用于存储每次训练结果（标签）
    test_label_history = []  # 用于存储每次测试结果（标签）
    time_train_history = []  # 用于存储训练的时间
    t_total = time.time()

    for epoch in range(args.epochs):
        t = time.time()
        model.train()  # model.train()是保证BN(Batch Normalization）层用每一批数据的均值和方差
        optimizer.zero_grad()
        output = model(features, adjacency, prior_prob_tensor)  # 前向传播
        train_loss = F.nll_loss(output[idx_train], labels[idx_train])
        train_acc = accuracy(output[idx_train], labels[idx_train])
        train_lab = output[idx_train].max(1)[1]
        train_loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adjacency, prior_prob_tensor)

        val_loss = F.nll_loss(output[idx_val], labels[idx_val])
        val_acc = accuracy(output[idx_val], labels[idx_val])
        val_lab = output[idx_val].max(1)[1]
        test_loss, test_acc, test_lab = test(prior_prob_tensor)
        print('Epoch: {:03d}'.format(epoch + 1),
              'loss_train: {:.2f}'.format(train_loss.item()),
              'acc_train: {:.2f}'.format(train_acc.item()),
              'loss_val: {:.2f}'.format(val_loss.item()),
              'acc_val: {:.2f}'.format(val_acc.item()),
              'loss_test: {:.2f}'.format(test_loss.item()),
              'acc_test: {:.2f}'.format(test_acc.item()),
              'time: {:.2f}s'.format(time.time() - t))
        # 记录训练过程中损失值和准确率的变化，用于画图
        train_loss_history.append(train_loss.item())
        train_acc_history.append(train_acc.item())
        train_label_history.append(train_lab)
        val_loss_history.append(val_loss.item())
        val_acc_history.append(val_acc.item())
        val_label_history.append(val_lab)
        test_loss_history.append(test_loss.item())
        test_acc_history.append(test_acc.item())
        test_label_history.append(test_lab)
        time_train_history.append(time.time() - t)
    print("总时间 {:.2f}".format(time.time() - t_total))
    return train_loss_history, train_acc_history, val_loss_history, \
        val_acc_history, val_label_history, train_label_history, test_loss_history, \
        test_acc_history, test_label_history,  time.time() - t_total


def test(prior_prob_tensor):
    # test_label_history = []
    model.eval()
    output = model(features, adjacency, prior_prob_tensor)
    test_loss = F.nll_loss(output[idx_test], labels[idx_test])
    test_acc = accuracy(output[idx_test], labels[idx_test])
    test_lab = output[idx_test].max(1)[1]
    # test_label_history.append(test_lab)
    # print("测试集结果:",
    #       "loss= {:.2f}".format(test_loss.item()),
    #       "accuracy= {:.2f}".format(test_acc.item()))
    return test_loss, test_acc, test_lab


# 加载预测数据
def load_predict_data(d_id, unk_k_nei, k_nei_id):
    features_predict = sp.csr_matrix(unk_k_nei, dtype=np.float32)
    edges_pre = []
    for k in range(len(k_nei_id)):
        for g in range(len(k_nei_id[k])):
            gg = [k, int(k_nei_id[k][g] - 1)]
            edges_pre.append(gg)
    edges_pre = np.array(edges_pre)

    adj_predict = sp.coo_matrix((np.ones(edges_pre.shape[0]), (edges_pre[:, 0], edges_pre[:, 1])),
                                shape=(len(d_id), len(d_id)), dtype=np.float32)
    adj_predict = adj_predict + \
        adj_predict.T.multiply(adj_predict.T > adj_predict) - \
        adj_predict.multiply(adj_predict.T > adj_predict)
    features_predict = normalize(features_predict)
    adj_predict = normalize(adj_predict + sp.eye(adj_predict.shape[0]))

    idx_predict = range(len(d_id))
    features_predict = torch.FloatTensor(np.array(features_predict.todense()))
    adj_predict = sparse_mx_to_torch_sparse_tensor(adj_predict)
    idx_predict = torch.LongTensor(idx_predict)

    if args.cuda:
        model.cuda()
        features_predict = features_predict.cuda()
        adj_predict = adj_predict.cuda()
        idx_predict = idx_predict.cuda()

    return idx_predict, features_predict, adj_predict


# 预测
def predict_(prior_pro_tensor):
    predict_results = []  # 用于存储预测结果的标签（0-1）
    pre_r = []  # 暂时存储预测结果的标签（0-1）
    for i in range(len(unknown_k_neighbor)):
        idx_predict, features_predict, adj_predict = \
            load_predict_data(data_id, unknown_k_neighbor[i], known_neighbor_id[i])
        pre_result = predict(features_predict, adj_predict, prior_pro_tensor)
        dd = []
        for s in range(pre_result.shape[0]):
            dd.append(pre_result[s].item())
        pre_r.append(dd)
        print("第{:}次,预测超标数量：{:},预测数量：{:}".format(i + 1, sum(dd), pre_result.shape[0]))
    for j in range(len(pre_r)):
        for n in range(len(pre_r[j])):
            predict_results.append(pre_r[j][n])
    print("超标点位个数", sum(predict_results))
    print("---预测数据整理完毕！")
    return predict_results


# 预测方法
def predict(features_pre, adjacency_pre, pre_prior_prob):
    model.eval()
    output = model(features_pre, adjacency_pre, pre_prior_prob)
    predict_lab = output.max(1)[1].type_as(labels)
    return predict_lab


# 预测结果图表化展示
def predict_result_accuracy(train_loss_data, train_accuracy_data, val_loss_data, val_accuracy_data,
                            test_loss_data, test_accuracy_data):
    x_data = np.arange(0, len(train_loss_data), 1)
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    lrs = args.lr * 1000
    drops = args.dropout * 10
    plt.subplot(1, 2, 1)
    plt.plot(x_data, train_loss_data)
    plt.plot(x_data, val_loss_data)
    plt.plot(x_data, test_loss_data)
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.title('The loss of %s' % b_name + '_NB_GCN_%d_%d_%d_%d_%d_%d' % (
        lrs, args.seed, drops, w_decay, cell_size_l, cell_size_v))
    plt.legend(['Training set', 'Validation set', 'test set'])

    plt.subplot(1, 2, 2)
    plt.plot(x_data, train_accuracy_data)
    plt.plot(x_data, val_accuracy_data)
    plt.plot(x_data, test_accuracy_data)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.title('The accuracy of %s' % b_name + '_NB_GCN_%d_%d_%d_%d_%d_%d' % (
        lrs, args.seed, drops, w_decay, cell_size_l, cell_size_v))
    plt.legend(['Training set', 'Validation set', 'test set'])
    plt.show()


# 导出结果
def result_output(pre_p_l, accuracy_tr, accuracy_v, accuracy_te,
                  loss_tr, loss_v, loss_te, s_path, label_v, label_tr, label_te, t_time):
    """
    导出结果
    :param pre_p_l: 最终预测集的预测标签
    :param accuracy_tr: 训练集的精度数据
    :param accuracy_v: 验证集的精度数据
    :param accuracy_te: 验证集的精度数据
    :param loss_tr: 训练集的损失值数据
    :param loss_v: 验证集的损害值数据
    :param loss_te: 验证集的损害值数据
    :param s_path: 文件保存路径
    :param label_v: 验证集的预测标签
    :param label_tr: 训练集的预测标签
    :param label_te: 训练集的预测标签
    :param t_time: 达到目标精度后，训练/验证所花的总时间
    :return:
    """
    workbook = xlsxwriter.Workbook(s_path)
    worksheet = workbook.add_worksheet("预测结果")
    worksheet_index = workbook.add_worksheet("预测结果评价指标")
    worksheet_label = workbook.add_worksheet("验证数据集的验证结果(标签)")
    worksheet_label_train = workbook.add_worksheet("训练数据集的验证结果(标签)")
    worksheet_label_test = workbook.add_worksheet("测试数据集的验证结果(标签)")
    # 第一个表——预测结果
    worksheet.write(0, 0, "待插值点位ID")
    worksheet.write(0, 1, "待插值点的坐标x")
    worksheet.write(0, 2, "待插值点的坐标y")
    worksheet.write(0, 3, "待插值点的坐标z")
    worksheet.write(0, 4, "待插值点的预测标签")
    for po in range(len(unknown_point_0)):
        worksheet.write(po + 1, 0, po + 1)
        worksheet.write(po + 1, 1, unknown_point_0[po][0])
        worksheet.write(po + 1, 2, unknown_point_0[po][1])
        worksheet.write(po + 1, 3, unknown_point_0[po][2])
        worksheet.write(po + 1, 4, 0)
    for p in range(len(unknown_point_list)):
        worksheet.write(p + len(unknown_point_0) + 1, 0, p + len(unknown_point_0) + 1)
        worksheet.write(p + len(unknown_point_0) + 1, 1, unknown_point_list[p][0])
        worksheet.write(p + len(unknown_point_0) + 1, 2, unknown_point_list[p][1])
        worksheet.write(p + len(unknown_point_0) + 1, 3, unknown_point_list[p][2])
        worksheet.write(p + len(unknown_point_0) + 1, 4, pre_p_l[p])

    # 第二个表——预测结果评价指标
    worksheet_index.write(0, 0, "训练的次数")
    worksheet_index.write(0, 1, "训练集精度")
    worksheet_index.write(0, 2, "验证集精度")
    worksheet_index.write(0, 3, "测试集精度")

    worksheet_index.write(0, 4, "训练集交叉熵损失")
    worksheet_index.write(0, 5, "验证集交叉熵损失")
    worksheet_index.write(0, 6, "测试集交叉熵损失")

    worksheet_index.write(0, 7, "训练集平均精度")
    worksheet_index.write(0, 8, "验证集平均精度")
    worksheet_index.write(0, 9, "测试集平均精度")

    worksheet_index.write(0, 10, "训练集的平均交叉熵损失")
    worksheet_index.write(0, 11, "验证集的平均交叉熵损失")
    worksheet_index.write(0, 12, "测试集的平均交叉熵损失")
    worksheet_index.write(0, 13, "达到目标精度后训练/验证所花的总时间")
    for n in range(len(accuracy_tr)):
        worksheet_index.write(n + 1, 0, n + 1)
        worksheet_index.write(n + 1, 1, accuracy_tr[n])
        worksheet_index.write(n + 1, 2, accuracy_v[n])
        worksheet_index.write(n + 1, 3, accuracy_te[n])
        worksheet_index.write(n + 1, 4, loss_tr[n])
        worksheet_index.write(n + 1, 5, loss_v[n])
        worksheet_index.write(n + 1, 6, loss_te[n])

    worksheet_index.write(1, 7, sum(accuracy_tr) / len(accuracy_tr))
    worksheet_index.write(1, 8, sum(accuracy_v) / len(accuracy_v))
    worksheet_index.write(1, 9, sum(accuracy_te) / len(accuracy_te))

    worksheet_index.write(1, 10, sum(loss_tr) / len(loss_tr))
    worksheet_index.write(1, 11, sum(loss_v) / len(loss_v))
    worksheet_index.write(1, 12, sum(loss_te) / len(loss_te))
    worksheet_index.write(1, 13, t_time)

    # 第三个表——验证数据集的验证结果(标签)
    val_o_label = labels[idx_val]
    worksheet_label.write(0, 0, "验证集的ID")
    worksheet_label.write(0, 1, "验证集的坐标x")
    worksheet_label.write(0, 2, "验证集的坐标y")
    worksheet_label.write(0, 3, "验证集的坐标z")
    worksheet_label.write(0, 4, "验证集的原始标签")
    for l_index in range(len(idx_val)):
        worksheet_label.write(l_index + 1, 0, idx_val[l_index])
        worksheet_label.write(l_index + 1, 1, original_data[idx_val[l_index]][0])
        worksheet_label.write(l_index + 1, 2, original_data[idx_val[l_index]][1])
        worksheet_label.write(l_index + 1, 3, original_data[idx_val[l_index]][2])
        worksheet_label.write(l_index + 1, 4, val_o_label[l_index])
    for v_index in range(len(label_v)):
        worksheet_label.write(0, v_index + 5, '第%d' % (v_index + 1) + '次验证预测标签')  # 标题
        for l_ind in range(len(idx_val)):
            worksheet_label.write(l_ind + 1, v_index + 5, label_v[v_index][l_ind])  # 内容

    # 第四个表——训练数据集的验证结果(标签)
    train_o_label = labels[idx_train]
    worksheet_label_train.write(0, 0, "训练集的ID")
    worksheet_label_train.write(0, 1, "训练集的坐标x")
    worksheet_label_train.write(0, 2, "训练集的坐标y")
    worksheet_label_train.write(0, 3, "训练集的坐标z")
    worksheet_label_train.write(0, 4, "训练集的原始标签")
    for i_index in range(len(idx_train)):
        worksheet_label_train.write(i_index + 1, 0, idx_train[i_index])
        worksheet_label_train.write(i_index + 1, 1, original_data[idx_train[i_index]][0])
        worksheet_label_train.write(i_index + 1, 2, original_data[idx_train[i_index]][1])
        worksheet_label_train.write(i_index + 1, 3, original_data[idx_train[i_index]][2])
        worksheet_label_train.write(i_index + 1, 4, train_o_label[i_index])
    for t_index in range(len(label_tr)):
        worksheet_label_train.write(0, t_index + 5, '第%d' % (t_index + 1) + '次训练预测标签')
        for i_index_v in range(len(idx_train)):
            worksheet_label_train.write(i_index_v + 1, t_index + 5, label_tr[t_index][i_index_v])

    # 第五个表——测试数据集的验证结果(标签)
    train_o_label = labels[idx_test]
    worksheet_label_test.write(0, 0, "测试集的ID")
    worksheet_label_test.write(0, 1, "测试集的坐标x")
    worksheet_label_test.write(0, 2, "测试集的坐标y")
    worksheet_label_test.write(0, 3, "测试集的坐标z")
    worksheet_label_test.write(0, 4, "测试集的原始标签")
    for te_index in range(len(idx_test)):
        worksheet_label_test.write(te_index + 1, 0, idx_test[te_index])
        worksheet_label_test.write(te_index + 1, 1, original_data[idx_test[te_index]][0])
        worksheet_label_test.write(te_index + 1, 2, original_data[idx_test[te_index]][1])
        worksheet_label_test.write(te_index + 1, 3, original_data[idx_test[te_index]][2])
        worksheet_label_test.write(te_index + 1, 4, train_o_label[te_index])
    for tes_index in range(len(label_te)):
        worksheet_label_test.write(0, tes_index + 5, '第%d' % (tes_index + 1) + '次测试预测标签')
        for t_index_v in range(len(idx_test)):
            worksheet_label_test.write(t_index_v + 1, tes_index + 5, label_te[tes_index][t_index_v])
    workbook.close()
    print("预测结果数据导出成功！")


if __name__ == "__main__":
    # 计算先验概率
    # data_prior_probability = np.array(prior_probability()).astype(float)
    # 计算先验概率（考虑土壤因素）
    data_prior_probability = np.array(soil_prior_probability()).astype(float)
    # 先验概率数组转张量
    prior_probability_tensor = torch.Tensor(np.array(data_prior_probability)).cuda()
    # 训练
    loss_train, accuracy_train, loss_val, accuracy_val, val_label, train_label, loss_test, \
        accuracy_test, test_label, total_time = train(prior_probability_tensor)
    print("m_loss_train：{:.2f}, ".format(sum(loss_train) / len(loss_train)),
          "m_accuracy_train：{:.2f}, ".format(sum(accuracy_train) / len(accuracy_train)),
          "m_loss_val：{:.2f}, ".format(sum(loss_val) / len(loss_val)),
          "m_accuracy_val：{:.2f}, ".format(sum(accuracy_val) / len(accuracy_val)),
          "m_loss_test：{:.2f}, ".format(sum(loss_test) / len(loss_test)),
          "m_accuracy_test：{:.2f}".format(sum(accuracy_test) / len(accuracy_test)))
    # 预测结果图表化展示
    predict_result_accuracy(loss_train, accuracy_train, loss_val, accuracy_val, loss_test, accuracy_test)

    # 预测
    pre_po_label = predict_(prior_probability_tensor)

    # 数据导出
    lr = args.lr * 1000
    drop = args.dropout * 10

    # save_path = "..\\predict_result3\\BaP_NB_GCN_%d_%d_%d_%d" % (lr, args.seed, drop, w_decay) + ".xlsx"
    # save_path = "..\\predict_result3\\DBA_NB_GCN_%d_%d_%d_%d" % (lr, args.seed, drop, w_decay) + ".xlsx"
    # save_path = "..\\predict_result4\\BaP_NB_GCN_%d_%d_%d_%d_%d_%d_%d" % (lr, args.seed, drop, w_decay,
    #                                                                    cell_size_l, cell_size_v, t_value) + ".xlsx"
    save_path = "result\\DBA_NB_GCN_%d_%d_%d_%d_%d_%d_%d" % (lr, args.seed, drop, w_decay,
                                                                       cell_size_l, cell_size_v, t_value) + ".xlsx"
    result_output(pre_po_label, accuracy_train, accuracy_val, accuracy_test,
                  loss_train, loss_val, loss_test, save_path,
                  val_label, train_label, test_label, total_time)
    torch.cuda.empty_cache()
