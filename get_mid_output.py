import os

import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


from resnet_18 import ResNet18
from resnet_50 import ResNet50


ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

BATCH_SIZE = 256


# 钩子类
class Hook():

    def __init__(
        self,
        model_depth,
        model_mode
    ):
        self.stage1_list = []
        self.stage4_list = []
        self.model_mode = model_mode
        self.model_depth = model_depth
    
    def hook_1(self, module, input, output):

        self.stage1_list.append(
            output.detach().cpu().numpy()
        )

    def hook_4(self, module, input, output):

        self.stage4_list.append(
            output.detach().cpu().numpy()
        )
    
    def to_dict(self):

        return {
            'stage1': np.concatenate(
                self.stage1_list,
                axis = 0
            ),

            'stage4': np.concatenate(
                self.stage4_list,
                axis = 0
            )
        }


# 获取中间层输出函数
def get_mid_output(
    model_depth,
    model_mode, 
    test_loader
):
    
    # 构造模型
    if model_depth == 18:

        if model_mode == 'FULL':
            
            model = ResNet18(
                res_connection = True
            )
        
        elif model_mode == 'Nres':

            model = ResNet18(
                res_connection = False
            )
    
    elif model_depth == 50:

        if model_mode == 'FULL':

            model = ResNet50(
                res_connection = True
            )

        elif model_mode == 'Nres':

            model = ResNet50(
                res_connection = False
            )
    
    # 加载模型参数
    model.load_state_dict(
        torch.load(
            f'{ROOT}/models_params/' + 
            f'ResNet{model_depth}' +
            f'_{model_mode}.pth'
        )
    )

    # 标签列表
    label_list = []

    # 构造钩子对象
    hook = Hook(
        model_mode = model_mode,
        model_depth = model_depth,
    )

    # 注册钩子
    model.stage1.register_forward_hook(hook.hook_1)
    model.stage4.register_forward_hook(hook.hook_4)

    # 模型加载到GPU
    model = model.cuda()

    # 模型评估模式
    model.eval()
 
    # 推理测试集
    with torch.no_grad():

        for i, (inputs, label) in enumerate(
            tqdm(
                iterable = test_loader,
                desc = 'Getting mid output'
            )
        ):
            
            inputs = inputs.cuda()
            label = label.cuda()
            
            model(inputs)
    
            label_list.append(label)

    labels = torch.cat(label_list, dim=0).cpu().numpy()

    # 返回中间输出与标签
    return hook.to_dict() , labels


if __name__ == '__main__':

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

    testset = CIFAR10(
        root = f'{ROOT}/data',
        train = False,
        transform = transform,
        download = True
    )

    subset = Subset(testset, list(range(500)))

    testloader = DataLoader(
        subset,
        batch_size = BATCH_SIZE,
        shuffle = False,
    )

    for model in [
        (50, 'FULL'),
        (50, 'Nres'),
        (18, 'FULL'),
        (18, 'Nres'),
    ]:
        
        torch.cuda.empty_cache()

        mid_output, labels = get_mid_output(
            model_depth = model[0],
            model_mode = model[1],
            test_loader = testloader
        )

        print(labels)

        with h5py.File(
            name = f'{ROOT}/model_data/' + 
                    f'mid_output.h5', 
            mode = 'a'
        ) as file:
            
            group_name = f'{model[0]}_{model[1]}'
            
            group = file.create_group(group_name)
            
            group.create_dataset(
                name = 'stage1',
                data = mid_output['stage1']
            )
            
            group.create_dataset(
                name = 'stage4',
                data = mid_output['stage4']
            )

            group.create_dataset(
                name = 'labels',
                data = labels
            )