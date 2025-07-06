import os
import sys
import argparse
import logging
from datetime import datetime
from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)


class DualOutputHandler:
    """同时输出到终端和日志文件，保持终端格式不变"""

    def __init__(self, terminal_stream, file_handler):
        self.terminal = terminal_stream
        self.file_handler = file_handler

    def write(self, message):
        self.terminal.write(message)
        if message.strip():  # 避免记录空行
            self.file_handler.emit(logging.LogRecord(
                name="STDOUT",
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=message.rstrip(),
                args=None,
                exc_info=None
            ))

    def flush(self):
        self.terminal.flush()


def setup_logging(result_dir):
    """配置日志系统（保持终端输出不变）"""
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 基础日志配置（仅文件）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file)],
        # force=True
    )
    logger = logging.getLogger(__name__)

    # 重定向stdout（保持终端原始输出）
    original_stdout = sys.stdout
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    sys.stdout = DualOutputHandler(original_stdout, file_handler)

    # 配置tqdm（如果存在）
    if 'tqdm' in sys.modules:
        from tqdm import tqdm
        original_tqdm_write = tqdm.write

        def tqdm_write_hook(msg, file=None, **kwargs):
            original_tqdm_write(msg, file=file, **kwargs)
            logger.info(msg.strip())

        tqdm.write = tqdm_write_hook

    logger.info(f"日志文件已创建: {log_file}")
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/home/sunhao/DLB/DiffUIR/data')
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--direction', type=str, default='AtoB')
    parser.add_argument('--preprocess', type=str, default='none')
    parser.add_argument('--no_flip', type=bool, default=True)
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--result_dir", type=str, default='/home/sunhao/DLB/DiffUIR/results_out1')
    return parser.parse_args()


def main():
    # 初始化设置
    set_seed(10)
    opt = parse_args()
    logger = setup_logging(opt.result_dir)

    # 参数设置
    config = {
        'save_and_sample_every': 1000,
        'sampling_timesteps': int(sys.argv[1]) if len(sys.argv) > 1 else 5,
        'train_num_steps': 100000,
        'condition': True,
        'train_batch_size': 1,
        'num_samples': 1,
        'image_size': 256
    }

    logger.info("程序启动配置:\n" + "\n".join(f"{k}: {v}" for k, v in {**vars(opt), **config}.items()))

    try:
        # 数据集加载
        dataset = AlignedDataset_all(
            opt, config['image_size'],
            augment_flip=False,
            equalizeHist=True,
            crop_patch=False,
            generation=False,
            task='rain'
        )

        # 模型初始化
        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=1,
            condition=True,
            objective='pred_res',
            test_res_or_noise="res"
        )

        diffusion = ResidualDiffusion(
            model,
            image_size=config['image_size'],
            timesteps=1000,
            delta_end=1.8e-3,
            sampling_timesteps=3,
            ddim_sampling_eta=0.,
            objective='pred_res',
            loss_type='l1',
            condition=True,
            sum_scale=0.01,
            test_res_or_noise="res",
        )

        trainer = Trainer(
            diffusion,
            dataset,
            opt,
            train_batch_size=config['train_batch_size'],
            num_samples=config['num_samples'],
            train_lr=2e-4,
            train_num_steps=config['train_num_steps'],
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=False,
            convert_image_to="RGB",
            results_folder="./ckpt_universal/diffuir_all",
            condition=True,
            save_and_sample_every=config['save_and_sample_every'],
            num_unet=1,
        )

        # 测试流程
        if trainer.accelerator.is_local_main_process:
            print("\n=== 开始测试 ===")  # 终端显示特殊格式
            trainer.load(130)
            trainer.set_results_folder(opt.result_dir)
            trainer.test(last=True, dataroot=opt.dataroot)
            print("=== 测试完成 ===\n")  # 终端显示特殊格式

    except Exception as e:
        logger.exception("程序异常终止")
        raise
    finally:
        logger.info("程序运行结束")
        sys.stdout = sys.__stdout__  # 恢复原始stdout


if __name__ == "__main__":
    main()