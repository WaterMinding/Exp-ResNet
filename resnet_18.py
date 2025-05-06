# 导入第三方库模块
import torch
import torch.nn as nn

# 残差块类
class BasicBlock(nn.Module):

    # 构造方法
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        bn = True,
        res_connection=True,
    ):
        
        # 调用父类构造方法
        super(BasicBlock, self).__init__()        

        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
        )

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

        # 激活函数
        self.relu = nn.ReLU()

        # 根据批归一化选择处理归一化层
        if not bn:
            
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn_ds = nn.Identity()
        
        else:

            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        
        # 保存残差连接选择
        self.res_connection = res_connection
        
        # 定义下采样
        self.downsample = None
        
        if stride != 1 or in_channels != out_channels:
        
            self.downsample = nn.Sequential(
        
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                ),

                self.bn_ds,
            )

    # 前向传播
    def forward(self, x):

        # 卷积层1
        out = self.conv1(x)

        # 批归一化层1
        out = self.bn1(out)

        # 激活函数
        out = self.relu(out)

        # 卷积层2
        out = self.conv2(out)

        # 批归一化层2
        out = self.bn2(out)

        # 残差连接
        if self.res_connection:

            # 下采样
            if self.downsample is not None:

                identity = self.downsample(x)
            
            else:

                identity = x

            out += identity
        
        # 激活函数
        out = self.relu(out)

        return out


# Stage 0
class Stage0(nn.Module):

    # 构造方法
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        bn = True
    ):

        # 调用父类构造方法
        super(Stage0, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 7,
            stride = 2,
            padding = 3,
        )

        # 批归一化层
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Sequential()

        # 激活函数
        self.relu = nn.ReLU()

        # 最大池化层
        self.maxpool = nn.MaxPool2d(
            kernel_size = 3, 
            stride = 2, 
            padding = 1
        )

    # 前向传播方法
    def forward(self, x):

        # 第一个卷积层
        out = self.conv1(x)

        # 批归一化层
        out = self.bn1(out)

        # 激活函数
        out = self.relu(out)

        # 最大池化层
        out = self.maxpool(out)

        return out
    

# Stage 1-4
class Stage(nn.Module):

    # 构造方法
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        bn = True,
        res_connection = True
    ):

        # 调用父类构造方法
        super(Stage, self).__init__()

        # 第一个残差块
        self.block1 = BasicBlock(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = stride,
            bn = bn,
            res_connection = res_connection,
        )

        # 第二个残差块
        self.block2 = BasicBlock(
            in_channels = out_channels,
            out_channels = out_channels,
            stride = 1,
            bn = bn,
            res_connection = res_connection,
        )

    # 前向传播方法
    def forward(self, x):

        # 第一个残差块
        out = self.block1(x)

        # 第二个残差块
        out = self.block2(out)

        return out


# ResNet-18
class ResNet18(nn.Module):

    # 构造方法
    def __init__(
        self,
        num_classes = 10,
        bn = True,
        res_connection = True
    ):
        
        # 调用父类构造方法
        super(ResNet18, self).__init__()

        # 阶段0
        self.stage0 = Stage0(
            in_channels = 3,
            out_channels = 64,
            bn = bn,
        )

        # 阶段1
        self.stage1 = Stage(
            in_channels = 64,
            out_channels = 64,
            stride = 1,
            bn = bn,
            res_connection = res_connection,
        )

        # 阶段2
        self.stage2 = Stage(
            in_channels = 64,
            out_channels = 128,
            stride = 2,
            bn = bn,
            res_connection = res_connection,
        )

        # 阶段3
        self.stage3 = Stage(
            in_channels = 128,
            out_channels = 256,
            stride = 2,
            bn = bn,
            res_connection = res_connection,
        )

        # 阶段4
        self.stage4 = Stage(
            in_channels = 256,
            out_channels = 512,
            stride = 2,
            bn = bn,
            res_connection = res_connection,
        )

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # 全连接层
        self.fc = nn.Linear(
            in_features = 512,
            out_features = num_classes
        )

    # 定义前向传播
    def forward(self, x):

        # 阶段0
        out:torch.Tensor = self.stage0(x)

        # 阶段1
        out = self.stage1(out)

        # 阶段2
        out = self.stage2(out)

        # 阶段3
        out = self.stage3(out)

        # 阶段4
        out = self.stage4(out)

        # 全局平均池化
        out = self.global_avg_pool(out)

        # 展平
        out = out.view(out.size(0), -1)
    
        # 全连接层
        out = self.fc(out)

        return out