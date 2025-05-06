# 导包
import torch
from torch import nn

# 残差块类
class BottleNeck(nn.Module):

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
        super(BottleNeck, self).__init__()

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=stride
        )

        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 定义第三个卷积层
        self.conv3 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

        # 定义下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:

            # 定义下采样层
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
            
        # 定义BatchNorm层
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels // 4)
            self.bn2 = nn.BatchNorm2d(out_channels // 4)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = self.bn2 = self.bn3 = self.bn_ds = None

        # 定义激活函数
        self.relu = nn.ReLU()

        # 保存残差连接选择
        self.res_connection = res_connection
    
    # 定义前向传播
    def forward(self, x):

        # 第一个卷积层
        h1 = self.conv1(x)

        # 如果有BatchNorm层，则进行BatchNorm处理
        if self.bn1 != None:
            h1 = self.bn1(h1)

        # 激活函数
        h1 = self.relu(h1)

        # 第二个卷积层
        h2 = self.conv2(h1)

        # 如果有BatchNorm层，则进行BatchNorm处理
        if self.bn2 != None:
            h2 = self.bn2(h2)

        # 激活函数
        h2 = self.relu(h2)

        # 第三个卷积层
        h3 = self.conv3(h2)

        # 如果有BatchNorm层，则进行BatchNorm处理
        if self.bn3 != None:
            h3 = self.bn3(h3)

        # 如果有残差连接，则将输入x与h3相加
        if self.res_connection:

            # 下采样
            if self.downsample != None:
                x = self.downsample(x)

                # 如果有BatchNorm层，则进行BatchNorm处理
                if self.bn_ds != None:
                    x = self.bn_ds(x)

            # 残差连接
            h3 = h3 + x

        # 激活函数
        output = self.relu(h3)

        return output


# 阶段〇类
class Stage0(nn.Module):

    # 构造方法
    def __init__(self, in_channels, out_channels, bn = True):

        # 调用父类构造方法
        super(Stage0, self).__init__()

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)

        # 定义MaxPooling层
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义激活函数
        self.relu = nn.ReLU()

        # 定义BatchNorm层
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    # 前向传播方法
    def forward(self, x):

        # 第一个卷积层
        h1 = self.conv1(x)

        # 如果有BatchNorm层，则进行BatchNorm处理
        if self.bn != None:
            h1 = self.bn(h1)

        # 激活函数
        h1 = self.relu(h1)

        # MaxPooling层
        output = self.pool(h1)

        return output


# 阶段类
class Stage(nn.Module):

    # 构造方法
    def __init__(
        self, 
        in_channels,
        out_channels,
        num_blocks,
        stride, 
        bn = True,
        res_connection = True
    ):
        
        # 调用父类构造方法
        super(Stage, self).__init__()

        # 定义第一个残差块
        self.res_block1 = BottleNeck(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = stride,
            bn = bn,
            res_connection = res_connection
        )

        # 定义剩余的残差块
        self.res_blocks = nn.ModuleList()
        
        for i in range(num_blocks - 1):
            self.res_blocks.append(
                BottleNeck(
                    in_channels = out_channels,
                    out_channels = out_channels,
                    stride = 1,
                    bn = bn,
                    res_connection = res_connection
                )
            )

    # 前向传播方法
    def forward(self, x):

        # 第一个残差块
        h1 = self.res_block1(x)

        # 剩余的残差块
        for res_block in self.res_blocks:
            h1 = res_block(h1)

        return h1
    

# ResNet-50网络
class ResNet50(nn.Module):

    # 构造方法
    def __init__(
        self, 
        num_classes = 10, 
        bn = True, 
        res_connection = True
    ):

        # 调用父类构造方法
        super(ResNet50, self).__init__()

        # 定义阶段0
        self.stage0 = Stage0(
            in_channels = 3,
            out_channels = 64,
            bn = bn
        )

        # 定义阶段1
        self.stage1 = Stage(
            in_channels = 64,
            out_channels = 256,
            num_blocks = 3,
            stride = 1,
            bn = bn,
            res_connection = res_connection
        )

        # 定义阶段2
        self.stage2 = Stage(
            in_channels = 256,
            out_channels = 512,
            num_blocks = 4,
            stride = 2,
            bn = bn,
            res_connection = res_connection
        )

        # 定义阶段3
        self.stage3 = Stage(
            in_channels = 512,
            out_channels = 1024,
            num_blocks = 6,
            stride = 2,
            bn = bn,
            res_connection = res_connection
        )

        # 定义阶段4
        self.stage4 = Stage(
            in_channels = 1024,
            out_channels = 2048,
            num_blocks = 3,
            stride = 2,
            bn = bn,
            res_connection = res_connection
        )

        # 定义全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 定义展平层
        self.flatten = nn.Flatten()

        # 定义全连接层
        self.fc = nn.Linear(2048, num_classes)

    # 前向传播方法
    def forward(self, x):

        # 阶段0
        h0 = self.stage0(x)

        # 阶段1
        h1 = self.stage1(h0)

        # 阶段2
        h2 = self.stage2(h1)

        # 阶段3
        h3 = self.stage3(h2)

        # 阶段4
        h4 = self.stage4(h3)

        # 全局平均池化
        h5 = self.global_avg_pool(h4)

        # 展平
        h6 = self.flatten(h5)

        # 全连接层
        output = self.fc(h6)

        return output


