import os
import sys
import argparse

from torch.utils.data import ConcatDataset

from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)
# 解析参数
def parsr_args():
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument("--dataroot", type=str, default='/home/sunhao/DLB/DiffUIR/data')
    # 训练阶段
    parser.add_argument("--phase", type=str, default='train')
    # 最大数据集大小
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    # 加载图像大小
    parser.add_argument('--load_size', type=int, default=268, help='scale images to this size') #572,268

    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    # 批量大小
    parser.add_argument("--bsize", type=int, default=2)
    # 返回参数
    opt = parser.parse_args()
    return opt

# 设置随机种子
sys.stdout.flush()
set_seed(10)

# 每1000步保存一次
save_and_sample_every = 1000
# 如果命令行参数中提供了采样步数，则使用该步数，否则使用默认步数10
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

# 训练批量大小
train_batch_size = 10
# 样本数量
num_samples = 1
# 尺度
sum_scale = 0.01
# 图像大小
image_size = 256
# 条件
condition = True
# 参数
opt = parsr_args()
# 结果文件夹
results_folder = "./ckpt_universal/diffuir_rain"

# 如果结果文件夹包含'universal'，则使用以下数据集
if 'universal' in results_folder:
    # 创建数据集
    dataset_fog = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='fog')
    dataset_light = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='light_only')
    dataset_rain = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='rain')
    dataset_snow = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='snow')
    dataset_blur = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='blur')
    # dataset = ConcatDataset([dataset_fog, dataset_light, dataset_rain, dataset_snow, dataset_blur])
    # dataset = ConcatDataset([dataset_fog, dataset_rain, dataset_snow, dataset_blur])
    # dataset = ConcatDataset([dataset_fog,dataset_light,dataset_rain, dataset_snow, dataset_blur])

    dataset = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='rain')
    # 模型数量
    num_unet = 1
    # 目标
    objective = 'pred_res'
    # 测试残差或噪声
    test_res_or_noise = "res"

    # 原数据步为300000
    # 训练步数（训练按步数结束）
    train_num_steps = 300000
    # 训练批量大小
    train_batch_size = 1
    # 尺度
    sum_scale = 0.01
    # 结束时间
    delta_end = 1.8e-3


# 创建模型
model = UnetRes(
    # 维度
    dim=64,
    # 维度乘数
    dim_mults=(1, 2, 4, 8),
    # 模型数量
    num_unet=num_unet,
    # 条件
    condition=condition,
    # 目标
    objective=objective,
    # 测试残差或噪声
    test_res_or_noise = test_res_or_noise
)
# model = UnetRes(
#     dim=32,
#     dim_mults=(1, 1, 1, 1),
#     num_unet=num_unet,
#     condition=condition,
#     objective=objective,
#     test_res_or_noise = test_res_or_noise
# )
# 创建扩散模型
diffusion = ResidualDiffusion(
    # 模型
    model,
    # 图像大小
    image_size=image_size,
    # 时间步数
    timesteps=1000,           # number of steps
    # 结束时间
    delta_end = delta_end,
    # 采样步数
    sampling_timesteps=sampling_timesteps,
    # 目标
    objective=objective,
    # 损失类型
    loss_type='l1',            # L1 or L2
    # 条件
    condition=condition,
    # 尺度
    sum_scale=sum_scale,
    # 测试残差或噪声
    test_res_or_noise = test_res_or_noise,
)

# 创建训练器
trainer = Trainer(
    # 扩散模型
    diffusion,
    # 数据集
    dataset,
    # 参数
    opt,
    # 训练批量大小
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    # 学习率
    train_lr=8e-5,
    # 训练步数
    train_num_steps=train_num_steps,         # total training steps
    # 梯度累积步数
    gradient_accumulate_every=2,    # gradient accumulation steps
    # 指数移动平均衰减
    ema_decay=0.995,                # exponential moving average decay
    # 混合精度
    amp=False,                        # turn on mixed precision
    # 转换图像
    convert_image_to="RGB",
    # 结果文件夹
    results_folder = results_folder,
    # 条件
    condition=condition,
    # 每1000步保存一次
    save_and_sample_every=save_and_sample_every,
    # 模型数量
    num_unet=num_unet,
)

if __name__ == '__main__':
    # train
    # trainer.load(30)
    trainer.train()
