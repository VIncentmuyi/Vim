import argparse
import copy
import os
import os.path as osp
import time
import gc  # 添加gc模块用于手动触发垃圾回收

from mmseg.datasets.builder import DATASETS
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
import mmcv_custom  # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmseg import __version__
from mmseg.apis import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

from backbone import vim


# 自定义内存清理函数
def clean_gpu_memory():
    """清理GPU内存缓存和触发垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 确保CUDA操作完成


# 修改train_segmentor函数以包含内存清理
def custom_train_segmentor(model, datasets, cfg, distributed=False, validate=True, timestamp=None, meta=None):
    """
    包装原始train_segmentor函数，在验证后添加内存清理
    """
    from mmcv_custom import train_segmentor as original_train_segmentor

    # 获取logger以便记录内存使用情况
    logger = get_root_logger()

    # 保存原始validate函数
    original_validate = cfg.get('validate', None)

    # 创建自定义validate函数，在每次验证后清理内存
    if validate and original_validate:
        original_eval_hook = cfg.get('evaluation', None)
        if original_eval_hook:
            original_after_eval = original_eval_hook.get('after_eval', None)

            def custom_after_eval(*args, **kwargs):
                # 执行原始after_eval函数(如果有)
                result = None
                if original_after_eval:
                    result = original_after_eval(*args, **kwargs)

                # 清理GPU内存并记录日志
                logger.info('Cleaning GPU memory after validation...')
                clean_gpu_memory()

                # 检查并记录当前内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    logger.info(f'GPU memory after cleanup: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB')

                return result

            # 替换原始after_eval函数
            cfg.evaluation.after_eval = custom_after_eval

    # 调用原始训练函数
    return original_train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--memory-monitor',
        action='store_true',
        help='whether to monitor GPU memory usage during training')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='override the batch size in config file')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.launcher == 'none':
        import os
        import torch.distributed as dist
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        print("已强制初始化分布式环境，使用单进程模式")
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 在开始前清理GPU内存
    clean_gpu_memory()

    # 重写批量大小（如果指定）
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
        print(f"Overriding batch size to {args.batch_size}")

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 如果启用了memory-monitor，添加内存使用情况监控
    if args.memory_monitor and torch.cuda.is_available():
        logger.info("Memory monitoring enabled")
        # 记录训练开始前的内存状态
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        logger.info(f'Initial GPU memory: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # 使用自定义训练函数代替原始函数
    custom_train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
