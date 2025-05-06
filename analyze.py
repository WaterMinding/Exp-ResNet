import os
import json

import h5py
import matplotlib
import numpy as np
from matplotlib.axes import Axes
from pandas import DataFrame as DF
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# 设置matplotlib正常显示中文与负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 分析ResNet网络训练准确率函数
# 参数1：net_depth：网络深度（18, 50）
def analyze_train_log(net_depth: int):
    
    figure = Figure(figsize=(25, 9), dpi=100)
    ax_train = figure.add_subplot(121)
    ax_test = figure.add_subplot(122)

    type_list = ['FULL', 'Nbn', 'Nres', 'Nboth']
    color_list = ['r', 'g', 'b', 'y']

    for net_type, color in zip(type_list, color_list):

        log_name = f'ResNet{net_depth}_{net_type}.json'

        with open(
            file = f'./model_data/{log_name}',
            mode = 'r'
        ) as log_file:
            
            log: dict = json.load(log_file)
    
            epochs = list(
                range(
                    1,
                    log['n_epochs'] + 1,
                )
            )

            train_acc = log['train_acc_list']
            test_acc = log['test_acc_list']
            
            ax_train.plot(
                epochs, 
                train_acc,
                f'{color}o--',
                label = f'ResNet{net_depth}_{net_type}'
            )

            ax_test.plot(
                epochs,
                test_acc,
                f'{color}o-',
                label = f'ResNet{net_depth}_{net_type}'
            )

    ax_train.set_ylim(0, 1.05)
    ax_test.set_ylim(0, 1.05)
    ax_train.set_xticks(epochs)
    ax_test.set_xticks(epochs)
    ax_train.set_xlabel('Epochs')
    ax_test.set_xlabel('Epochs')
    ax_train.set_ylabel('Train Accuracy')
    ax_test.set_ylabel('Test Accuracy')
    ax_train.set_title(f'ResNet{net_depth} 训练准确率')
    ax_test.set_title(f'ResNet{net_depth} 测试准确率')
    
    ax_train.legend()
    ax_test.legend()
    figure.savefig(
        f'{ROOT}/result/' + 
        f'ResNet{net_depth}_acc.png'
    )
    

# 分析 ResNet 网络训练梯度
def analyze_gradient(net_depth: int):
    
    figure = Figure(figsize=(25, 9))

    ax_shallow = figure.add_subplot(121)
    ax_deep = figure.add_subplot(122)

    type_list = ['FULL', 'Nbn', 'Nres', 'Nboth']
    color_list = ['r', 'g', 'b', 'y']

    for net_type, color in zip(type_list, color_list):

        log_name = f'ResNet{net_depth}_{net_type}.json'

        with open(
            file = f'./model_data/{log_name}', 
            mode = 'r'
        ) as log_file:
            
            log: dict = json.load(log_file)

            shallow_grad = log['shallow_grad_info_list']
            deep_grad = log['deep_grad_info_list']

            shallow_grad = np.array(shallow_grad)
            deep_grad = np.array(deep_grad)

            shallow_grad = np.log10(shallow_grad)
            deep_grad = np.log10(deep_grad)

            backwards_num = list(
                range(
                    len(shallow_grad)
                )
            )

            ax_shallow.scatter(
                backwards_num,
                shallow_grad,
                c = color,
                label = net_type
            )

            ax_deep.scatter(
                backwards_num,
                deep_grad,
                c = color,
                label = net_type
            )

    ax_shallow.set_xticks(
        range(1,len(backwards_num) - 1,500)
    )

    ax_deep.set_xticks(
        range(1,len(backwards_num) - 1,500)
    )

    ax_shallow.set_ylim(-30, 0)
    ax_deep.set_ylim(-30, 0)

    ax_shallow.set_xlabel('反向传播次数')
    ax_deep.set_xlabel('反向传播次数')
    ax_shallow.set_ylabel('阶段1梯度平均绝对值')
    ax_deep.set_ylabel('阶段4梯度平均绝对值')

    ax_shallow.set_title(
        f'ResNet{net_depth} ' + 
        '阶段1梯度平均绝对值与反向传播次数关系'
    )
    
    ax_deep.set_title(
        f'ResNet{net_depth} ' + 
        '阶段4梯度平均绝对值与反向传播次数关系'
    )

    ax_shallow.legend()
    ax_deep.legend()
    figure.savefig(
        f'{ROOT}/result/' + 
        f'ResNet{net_depth}_grad.png'
    )


# 分析深层与浅层输出特征图的可分性
def analyze_fisher(net_depth):

    type_list = ['FULL', 'Nres']
    stage_list = ['stage1', 'stage4']

    result_df = DF(
        {
            f'{net_depth}_FULL_stage1': [],
            f'{net_depth}_FULL_stage4': [],
            f'{net_depth}_Nres_stage1': [],
            f'{net_depth}_Nres_stage4': [],
        }
    )

    for net_type in type_list:

        for stage in stage_list:

            # 获取数据
            with h5py.File(
                f'{ROOT}/model_data/' +
                f'mid_output.h5',
                'r'
            ) as file:
                
                # stage 输出的特征图
                data = np.array(
                    file[
                        f'{net_depth}_{net_type}'
                    ][stage]
                )

                # 特征图对应的类别标签    
                labels = np.array(
                    file[
                        f'{net_depth}_{net_type}'
                    ][f'labels']
                )

            # 将数据按类别标签分类
            label_dict = {}

            for i in range(10):
            
                label_dict[i] = data[
                    labels == i
                ].copy()

                # 此处为了便于计算离散度矩阵，将数据展平
                # 维度1：样本量 * 通道数
                # 维度2：特征图面积
                label_dict[i] = np.reshape(
                    label_dict[i],
                    (
                        label_dict[i].shape[0] * label_dict[i].shape[1],
                        -1
                    )
                )
            
            # 计算总体均值
            mean = np.mean(
                
                # 同上，将数据展平
                np.reshape(
                    data,
                    (
                        data.shape[0] * data.shape[1],
                        -1
                    )
                ),
                
                axis=0
            )

            # 计算各类别均值
            mean_dict = {}

            for i in range(10):

                mean_dict[i] = np.mean(
                    label_dict[i],
                    axis=0
                )

            # 计算各类类内离散度矩阵
            sw_dict = {}

            for i in range(10):

                sw_dict[i] = np.dot(
                    (label_dict[i] - mean_dict[i]).T,
                    (label_dict[i] - mean_dict[i])
                )

            sw = np.sum(
                list(sw_dict.values()),
                axis=0
            )

            # 计算类间离散度矩阵
            sb_dict = {}

            for i in range(10):
                
                sb_dict[i] = len(label_dict[i]) * np.dot(
                    
                    # numpy一维数组无法转置为列向量
                    # 这里增加一个维度
                    np.expand_dims(
                        mean_dict[i] - mean, 
                        axis = 0
                    ).T,
                    
                    np.expand_dims(
                        mean_dict[i] - mean,
                        axis = 0
                    ),
                    
                )

            sb = np.sum(
                list(sb_dict.values()),
                axis=0
            )

            # 求 sw 与 sb 的迹
            sw_trace = np.trace(sw)
            sb_trace = np.trace(sb)

            # 计算可分性判据
            fisher = sb_trace / sw_trace

            result_df[
                f'{net_depth}_{net_type}_{stage}'
            ] = [fisher]
    
    figure = Figure()
    
    ax = figure.add_subplot(111)

    ax.bar(
        x = result_df.columns,
        height = result_df.values[0]
    )

    ax.set_title(
        f'ResNet{net_depth} ' + 
        'fisher可分性 柱状图'
    )
    ax.set_xlabel('stage')
    ax.set_ylabel('fisher')

    ax.set_ylim(
        0, 0.015
    )

    figure.savefig(
        f'{ROOT}/result/ResNet{net_depth}_fisher.png'
    )
    
if __name__ == "__main__":

    analyze_fisher(18)
    analyze_fisher(50)