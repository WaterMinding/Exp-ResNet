# 导入依赖包
print("导包...")
import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from resnet_50 import ResNet50
from resnet_18 import ResNet18
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# 定义训练过程梯度信息列表
shallow_grad_info_list = []
deep_grad_info_list = []

# 钩子函数
def hook_fn_shallow(module,grad_input,grad_output):

    global shallow_grad_info_list

    shallow_grad_info_list.append(
        torch.mean(
            torch.abs(
                grad_output[0]
            )
        ).item()
    )

def hook_fn_deep(module,grad_input,grad_output):

    global deep_grad_info_list

    deep_grad_info_list.append(
        torch.mean(
            torch.abs(
                grad_output[0]
            )
        ).item()
    )

# 程序入口
if __name__ == "__main__":

    print("处理参数...")

    # 定义参数解析器
    parser = argparse.ArgumentParser(description='Train a ResNet on CIFAR')

    # 增加命令行参数:残差连接开关
    parser.add_argument('-r', '--ResConnection',type = int, default = 1, help='是否启动残差连接')

    # 增加命令行参数:BatchNorm开关
    parser.add_argument('-b', '--BatchNorm',type = int, default = 1, help='是否启动BatchNorm')

    # 增加命令行参数:加载已存在的模型开关
    parser.add_argument('-l', '--LoadModel',type = int, default = 0, help='是否加载已存在的模型')

    # 增加命令行参数:模型名称
    parser.add_argument('-m', '--ModelName',type = str,default = None,help='模型名称')

    # 增加命令行参数:训练轮数
    parser.add_argument('-n', '--NEpoch',type = int, default = 20, help='训练轮数')

    # 增加命令行参数:batch size
    parser.add_argument('-s', '--BatchSize',type = int, default = 128, help='batch size')

    # 增加命令行参数:模型规模
    parser.add_argument('-d', '--Depth',type = int, default = 50, help='模型规模')

    # 增加命令行参数:设备
    parser.add_argument('-D', '--Device',type = str, default = 'cuda', choices = ['cuda', 'cpu'], help='训练设备')

    # 增加命令行参数:随机种子
    parser.add_argument('-S', '--Seed',type = int, default = 42, help='随机种子')

    # 解析命令行参数
    args = parser.parse_args()

    # 确定是否使用残差连接
    RESCONNECTION = bool(args.ResConnection)

    # 确定 BatchNorm 是否启动
    BATCHNORM = bool(args.BatchNorm)

    # 确定是否加载已存在的模型
    LOADMODEL = bool(args.LoadModel)

    # 确定模型名称
    MODELNAME = args.ModelName

    # 确定训练轮数
    N_EPOCH = int(args.NEpoch)

    # 确定batch size
    BATCHSIZE = int(args.BatchSize)

    # 确定模型规模
    DEPTH = int(args.Depth)

    # 确定训练过程中随机行为种子
    RANDOM_SEED = int(args.Seed)
    print("随机种子",RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("设备：", device)
    
    # 设置学习率
    LR = 0.1

    # 确定文件路径
    ROOT = os.path.abspath(__file__)
    ROOT = os.path.dirname(ROOT)

    print("构造模型...")
    # 构造模型
    if not LOADMODEL:

        if DEPTH == 50:
        
            model = ResNet50(
                num_classes = 10, 
                res_connection = RESCONNECTION, 
                bn = BATCHNORM
            ).to(device)

        else:

            model = ResNet18(
                num_classes = 10,
                res_connection = RESCONNECTION,
                bn = BATCHNORM
            ).to(device)

    else:

        if DEPTH == 50:
        
            model = ResNet50()
            model.load_state_dict(
                torch.load(ROOT + "/models_params/" + f"{MODELNAME}.pth")
            )
        
        else:
            
            model = ResNet18()
            model.load_state_dict(
                torch.load(ROOT + "/models_params/" + f"{MODELNAME}.pth")
            )

    # 设置数据集转换方法
    transform = transforms.Compose([                        
        
        # 调整图像尺寸为224x224
        transforms.Resize((224,224)),
        
        # 将图像转换为张量
        transforms.ToTensor(),
        
        # 对图像进行归一化
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5)
        )
    ])

    print("加载数据...")
    # 加载数据集
    trainset = CIFAR10(
        root = f'{ROOT}/data',
        train = True,
        transform = transform,
        download = True
    )

    testset = CIFAR10(
        root = f'{ROOT}/data',
        train = False,
        transform = transform,
        download = True
    )

    # 更改数据标签为独热编码
    trainset.targets = torch.LongTensor(trainset.targets)
    testset.targets = torch.LongTensor(testset.targets)
    trainset.targets = F.one_hot(trainset.targets,10).float()
    testset.targets = F.one_hot(testset.targets,10).float()

    # 构造数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size = BATCHSIZE,
        shuffle = True,
    )

    testloader = DataLoader(
        testset,
        batch_size = BATCHSIZE,
        shuffle = False,
    )

    print("准备训练...")
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr = LR,
    )

    # 定义训练集准确率列表
    train_acc_list = []

    # 定义测试集准确率列表
    test_acc_list = []

    # 注册钩子函数，在训练过程中记录参数的梯度
    model.stage0.register_full_backward_hook(hook_fn_shallow)
    model.stage4.register_full_backward_hook(hook_fn_deep)

    # 训练模型
    print('训练开始...')
    for epoch in range(N_EPOCH):

        # 设置模型为训练模式
        model.train()

        # 定义训练正确率
        train_correct = 0

        # 从训练数据中随机抽取一个批次
        for i, (inputs, labels) in enumerate(
            tqdm(
                iterable = trainloader, 
                desc = f'Epoch {epoch + 1}/{N_EPOCH}'
            )
        ):

            # 将输入数据和标签转移到GPU上
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 编码预测结果
            predicted_indices = torch.argmax(outputs, dim = 1)
            predicted_indices = F.one_hot(
                predicted_indices, 
                num_classes = 10
            ).float()

            # 统计训练准确率
            train_correct += torch.all(
                labels == predicted_indices,
                dim = 1
            ).sum().item()

            # 计算训练集准确率
            train_accuracy = train_correct / len(trainloader.dataset)
            train_accuracy = round(train_accuracy, 4)

        # 打印训练结果信息
        print(
            f'损失[{loss.item()}] 训练集准确率[{train_accuracy}]'
        )

        # 每1轮训练，保存一次训练集的准确率
        if (epoch + 1) % 1 == 0:
            train_acc_list.append(train_accuracy)
        
        # 每1轮训练，进行一次测试
        if (epoch + 1) % 1 == 0:

            # 设置模型为评估模式
            model.eval()

            # 关闭梯度
            with torch.no_grad():

                # 定义测试正确率
                test_correct = 0

                # 从测试数据中随机抽取一个批次
                print()
                for i, (inputs, labels) in enumerate(
                    tqdm(
                        iterable = testloader,
                        desc = 'Testing ',
                    )
                ):

                    # 将输入数据和标签转移到GPU上
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = model(inputs)

                    # 编码预测结果
                    predicted_indices = torch.argmax(outputs, dim = 1)
                    predicted_indices = F.one_hot(
                        predicted_indices,
                        num_classes = 10
                    ).float()

                    # 统计测试准确率
                    test_correct += torch.all(
                        labels == predicted_indices,
                        dim = 1
                    ).sum().item()

                # 计算、保存并打印测试准确率
                test_accuracy = test_correct / len(testloader.dataset)
                test_accuracy = round(test_accuracy, 4)
                test_acc_list.append(test_accuracy)
                print("测试准确率" + str(test_accuracy * 100) + "%\n" )

    # 保存模型参数
    print('\n保存模型参数...')
    if MODELNAME != None:
        torch.save(model.state_dict(), f'{ROOT}/models_params/' + MODELNAME + '.pth')
    else:
        torch.save(model.state_dict(), f"{ROOT}/models_params/ResNet{DEPTH}.pth")
    
    # 保存测试参数和结果信息
    data_dict = {
        "random_seed": RANDOM_SEED,
        "n_epochs": N_EPOCH,
        "batch_size": BATCHSIZE,
        "learning_rate": LR,
        "train_acc_list": train_acc_list,
        "test_acc_list": test_acc_list,
        "shallow_grad_info_list": shallow_grad_info_list,
        "deep_grad_info_list": deep_grad_info_list
    }

    with open(f"{ROOT}/model_data/{MODELNAME}_acc.json", "w") as f:
        json.dump(data_dict, f)

    print('保存完成!')
