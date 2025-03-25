# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import numpy as np
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# if 'pos_embed' in state_dict:
def interpolate_pos_embed(model, state_dict):
    # 检查位置编码是否存在
    if 'pos_embed' not in state_dict:
        print("预训练模型中没有位置编码(pos_embed)，跳过位置编码插值")
        return

    try:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # 应该是192

        # 获取当前模型位置编码的形状
        pos_embed_current = model.pos_embed  # 形状为 [1, 1024, 192]

        print(f"预训练模型位置编码形状: {pos_embed_checkpoint.shape}")
        print(f"当前模型位置编码形状: {pos_embed_current.shape}")

        # 对于ViT类模型，197通常是196个patch位置+1个class token
        # 1024是32x32=1024个patch位置

        # 假设第一个是class token
        num_extra_tokens = 1

        # 提取class token
        if pos_embed_checkpoint.shape[1] > 1:
            class_token = pos_embed_checkpoint[:, 0:num_extra_tokens, :]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:, :]
        else:
            class_token = None
            pos_tokens = pos_embed_checkpoint

        # 计算原始尺寸和目标尺寸
        src_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5 + 0.1)  # 添加小量避免浮点误差
        dst_size = int((pos_embed_current.shape[1] - num_extra_tokens) ** 0.5 + 0.1)

        print(f"预训练模型位置编码：{src_size}x{src_size}, 目标：{dst_size}x{dst_size}")

        # 强制重塑为14x14x192，即使形状不完全匹配
        # 197-1=196，或者近似为14*14=196
        try:
            pos_tokens = pos_tokens.reshape(-1, src_size, src_size, embedding_size)
        except RuntimeError:
            print(f"无法将位置编码重塑为 {src_size}x{src_size}，尝试强制重塑")
            # 确定最接近的尺寸
            total_tokens = pos_tokens.shape[1]
            actual_size = int(np.sqrt(total_tokens) + 0.5)  # 四舍五入
            print(f"检测到实际尺寸为: {actual_size}x{actual_size}")

            # 裁剪或填充到完美平方形
            if actual_size ** 2 > total_tokens:
                # 需要填充
                padding_needed = actual_size ** 2 - total_tokens
                print(f"填充 {padding_needed} 个令牌")
                padding = torch.zeros((pos_tokens.shape[0], padding_needed, embedding_size),
                                      device=pos_tokens.device, dtype=pos_tokens.dtype)
                pos_tokens = torch.cat([pos_tokens, padding], dim=1)
            elif actual_size ** 2 < total_tokens:
                # 需要裁剪
                print(f"裁剪到 {actual_size}x{actual_size}")
                pos_tokens = pos_tokens[:, :actual_size ** 2, :]

            src_size = actual_size
            pos_tokens = pos_tokens.reshape(-1, src_size, src_size, embedding_size)

        # 执行插值
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # [B, C, H, W]
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(dst_size, dst_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1)  # [B, H, W, C]
        pos_tokens = pos_tokens.reshape(1, dst_size * dst_size, embedding_size)

        # 如果有class token，则拼接回去
        if class_token is not None and num_extra_tokens > 0:
            # 检查当前模型是否也有class token
            if pos_embed_current.shape[1] > dst_size * dst_size:
                new_pos_embed = torch.cat([class_token, pos_tokens], dim=1)
            else:
                # 当前模型没有class token，只使用位置编码
                new_pos_embed = pos_tokens
        else:
            new_pos_embed = pos_tokens

        # 更新状态字典
        state_dict['pos_embed'] = new_pos_embed
        print(f"位置编码已插值，新形状: {new_pos_embed.shape}")

    except Exception as e:
        import traceback
        print(f"位置编码插值失败: {e}")
        print(traceback.format_exc())
        print("跳过位置编码插值，但继续加载其他参数")
        if 'pos_embed' in state_dict:
            del state_dict['pos_embed']  # 删除不兼容的位置编码，允许其他参数继续加载