import glob
import os

import torch
import tqdm
import time
from pcdet.utils import common_utils, commu_utils
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from .train_utils import save_checkpoint, checkpoint_state
from .optimization import build_optimizer, build_scheduler
from pcdet.utils.kd_utils import kd_forwad

def get_temperature_from_lr(optimizer, min_temp=2, max_temp=200.0):  #lr-softmax （2，200）
    """
    Calculate temperature T based on current learning rate.
    Args:
        optimizer: The optimizer containing learning rate info
        min_temp: Minimum temperature value (default: 0.1)
        max_temp: Maximum temperature value (default: 10.0)
    Returns:
        float: Temperature value T
    """
    if not hasattr(optimizer, 'param_groups'):
        return 1.0  # Default temperature if no learning rate info

    current_lr = optimizer.lr
    initial_lr = optimizer.param_groups[0].get('initial_lr', 0.003)

    ratio = current_lr / initial_lr
    if min_temp==max_temp and max_temp==0:
        temperature = 200.0 if ratio >= 0.01 else (2.0 if ratio >= 0.005 else 0.0)
    else:
        temperature = max_temp - (max_temp - min_temp) * ratio
    return temperature

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, teacher_model=None,
                    extra_optim=None, extra_lr_scheduler=None, teacher_model_2=None, teacher_num=None,
                    teacher_models=None, temperature=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    forward_func = getattr(kd_forwad, cfg.KD.get('FORWARD_FUNC', 'forward'))
    #import pdb;pdb.set_trace()
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        batch['temperature'] = temperature
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)
        if extra_lr_scheduler is not None:
            extra_lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            if extra_optim is not None:
                tb_log.add_scalar('meta_data/extra_lr', float(extra_optim.lr), accumulated_iter)

        model.train()

        loss, tb_dict, disp_dict = forward_func(
            model, teacher_model, batch, optimizer, extra_optim, optim_cfg, load_data_to_gpu,
            teacher_model_2=teacher_model_2, teacher_num=teacher_num, teacher_models=teacher_models
        )

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        if cur_it % 20 == 0:
            torch.cuda.empty_cache()

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_t': f'({data_time.avg:.1f})',
                'b_t': f'{batch_time.val:.1f}({batch_time.avg:.1f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_kd(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                   lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                   merge_all_iters_to_one_epoch=False, teacher_model=None, teacher_model_2=None, teacher_models=None, teacher_num=None):

    extra_optim = extra_lr_scheduler = None
    if optim_cfg.get('EXTRA_OPTIM', None) and optim_cfg.EXTRA_OPTIM.ENABLED:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            extra_optim = build_optimizer(model.module, optim_cfg.EXTRA_OPTIM)
        else:
            extra_optim = build_optimizer(model, optim_cfg.EXTRA_OPTIM)

        # last epoch is no matter for one cycle scheduler
        extra_lr_scheduler, _ = build_scheduler(
            extra_optim, total_iters_each_epoch=len(train_loader), total_epochs=total_epochs,
            last_epoch=-1, optim_cfg=optim_cfg.EXTRA_OPTIM
        )


    accumulated_iter = start_iter

    if teacher_model is not None and cfg.KD.get('TEACHER_MODE', None):
        getattr(teacher_model, cfg.KD.TEACHER_MODE)()
        #-------------tea_2------------------
        if teacher_num==2:
            getattr(teacher_model_2, cfg.KD.TEACHER_MODE)()

    if teacher_model is not None and cfg.KD.get('TEACHER_BN_MODE', None) == 'train':
        teacher_model.apply(common_utils.set_bn_train)
        # -------------tea_2------------------
        if teacher_num == 2:
            teacher_model_2.apply(common_utils.set_bn_train)
    # teacher_num=5 ---tea_5----
    if teacher_models is not None:
        for i, teacher in enumerate(teacher_models, start=2):  # 从2开始编号
            if teacher is not None:
                if cfg.KD.get('TEACHER_MODE', None):
                    getattr(teacher, cfg.KD.TEACHER_MODE)()
                if cfg.KD.get('TEACHER_BN_MODE', None) == 'train':
                    teacher.apply(common_utils.set_bn_train)


    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        if optim_cfg.OPTIMIZER == 'SegmentedOneCycle':
            for cur_epoch in tbar:
                if train_sampler is not None:
                    train_sampler.set_epoch(cur_epoch)

                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler
                else:
                    cur_scheduler = lr_scheduler
                interval = cur_epoch // optim_cfg.TTT
                Cycle = interval // 2

                if interval % 2 == 0 and (Cycle < optim_cfg.Cycle_num): 
                    accumulated_iter = train_one_epoch(  # 训练一个epoch
                        model, optimizer, train_loader, model_func,
                        lr_scheduler=cur_scheduler,
                        accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                        rank=rank, tbar=tbar, tb_log=tb_log,
                        leave_pbar=(cur_epoch + 1 == total_epochs),
                        total_it_each_epoch=total_it_each_epoch,
                        dataloader_iter=dataloader_iter,
                        teacher_model=teacher_model,
                        extra_optim=extra_optim,
                        extra_lr_scheduler=extra_lr_scheduler,
                        teacher_model_2=None,
                        teacher_num=teacher_num,
                    )
                elif interval % 2 == 1 and (Cycle < optim_cfg.Cycle_num):  
                    temperature = 0.02
                    accumulated_iter = train_one_epoch(  # 训练一个epoch
                        model, optimizer, train_loader, model_func,
                        lr_scheduler=cur_scheduler,
                        accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                        rank=rank, tbar=tbar, tb_log=tb_log,
                        leave_pbar=(cur_epoch + 1 == total_epochs),
                        total_it_each_epoch=total_it_each_epoch,
                        dataloader_iter=dataloader_iter,
                        teacher_model=teacher_model_2,
                        extra_optim=extra_optim,
                        extra_lr_scheduler=extra_lr_scheduler,
                        teacher_model_2=None,
                        teacher_num=teacher_num,
                    )
                else:
                    optim_cfg.TTT = 2*optim_cfg.TTT
                    lr_scheduler, lr_warmup_scheduler = build_scheduler(
                        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=optim_cfg.NUM_EPOCHS,
                        last_epoch=cur_epoch, optim_cfg=optim_cfg
                    )
                    if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                        cur_scheduler = lr_warmup_scheduler
                    else:
                        cur_scheduler = lr_scheduler
                    accumulated_iter = train_one_epoch(
                        model, optimizer, train_loader, model_func,
                        lr_scheduler=cur_scheduler,
                        accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                        rank=rank, tbar=tbar, tb_log=tb_log,
                        leave_pbar=(cur_epoch + 1 == total_epochs),
                        total_it_each_epoch=total_it_each_epoch,
                        dataloader_iter=dataloader_iter,
                        teacher_model=teacher_model,
                        extra_optim=extra_optim,
                        extra_lr_scheduler=extra_lr_scheduler,
                        teacher_model_2=None,
                        teacher_num=None,
                        temperature=0.0
                    )

                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)  # 保存新的检查点
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
        else:
            for cur_epoch in tbar:
                if train_sampler is not None:
                    train_sampler.set_epoch(cur_epoch)

                # train one epoch
                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler
                else:
                    cur_scheduler = lr_scheduler

                temperature = get_temperature_from_lr(optimizer, optim_cfg.T_range[0], optim_cfg.T_range[1])  # T0=2，T1=200

                accumulated_iter = train_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    teacher_model=teacher_model,
                    extra_optim=extra_optim,
                    extra_lr_scheduler=extra_lr_scheduler,
                    teacher_model_2=teacher_model_2,
                    teacher_num=teacher_num,
                    teacher_models=teacher_models,
                    temperature=temperature
                )

                # save trained model
                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)  # 保存新的检查点
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
