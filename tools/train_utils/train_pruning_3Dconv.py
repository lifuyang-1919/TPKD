import glob
import os


from torch.autograd import Variable
from torchvision import datasets, transforms

# from vgg import vgg
import numpy as np
import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from .optimization import build_optimizer, build_scheduler
from pcdet.models import load_data_to_gpu
import spconv.pytorch as spconv
from pcdet.models import build_network
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def train_pruning_3Dconv_weight(model, optimizer, train_loader, train_set, MODEL, cfg, args, optim_cfg, start_epoch, total_epochs, ckpt_save_dir, PRUNING2D, Random):
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
    # ----------------------  pruning ----------------------------------------------------------------

    dataloader_iter = iter(train_loader)
    batch = next(dataloader_iter)
    load_data_to_gpu(batch)

    print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            total += m.weight.data.shape[0]

    pruned = 0
    cfg_layer = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm1d):
            weight_copy = m.weight.data.clone()
            size = m.weight.data.shape[0]
            thre_index = int(size * (1-cfg.PRUNING3D.retain_ratio3D))
            weights_sorted, _ = torch.sort(weight_copy.abs())

            if Random:
                weight_copy = torch.rand_like(weight_copy)
                weights_sorted, _ = torch.sort(weight_copy.abs())
            thre = weights_sorted[thre_index]
            mask = weight_copy.abs().ge(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_layer.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, torch.nn.MaxPool2d):
            cfg_layer.append('M')

    pruned_ratio = pruned / total
    print('pruning ratio', pruned_ratio)
    print('Pre-processing Successful!')
    print(cfg_layer)

    # MODEL['BACKBONE_3D']['NUM_FILTERS'] = [int(x * cfg.PRUNING3D.retain_ratio) for x in MODEL['BACKBONE_3D']['NUM_FILTERS']]
    MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] = int(MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] * cfg.PRUNING3D.retain_ratio3D)
    MODEL['BACKBONE_3D']['WIDTH'] = MODEL['BACKBONE_3D']['WIDTH'] * cfg.PRUNING3D.retain_ratio3D
    if PRUNING2D:
        MODEL['BACKBONE_2D']['WIDTH'] = MODEL['BACKBONE_2D']['WIDTH'] * cfg.PRUNING2D.retain_ratio
        try:
            MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] = int(MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] * cfg.PRUNING2D.retain_ratio)
        except:
            print('MODEL[DENSE_HEAD][SHARED_CONV_CHANNEL] is not exist.')
    newmodel=build_network(model_cfg=MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    newmodel.cuda()
    # --------transfer weights-----------------------
    layer_id_in_cfg = 0
    start_mask = torch.ones(5)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, torch.nn.BatchNorm1d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            c = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, spconv.SparseConv3d) or isinstance(m0, spconv.SubMConv3d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # w = m0.weight.data[:, idx0, :, :, :].clone()
            if m0.weight.data.shape[-1] < len(idx0):
                w = m0.weight.data.clone()
            else:
                w = m0.weight.data[:, :, :, :, idx0].clone()
            w = w[idx1, :, :, :, :].clone()
            m1.weight.data = w.clone()
        else:
            m1 = m0

    # ------------------- save checkpoint ------------------------------------------------------
    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % start_epoch)
    save_checkpoint(
        checkpoint_state(newmodel, optimizer, start_epoch), filename=ckpt_name,
    )
    print("Build new model successfully")
    print(newmodel)
    print("-----------------------Sparse3D Conv Prune End!")
    return newmodel



def compute_rank(feature_map):
    """
    feature_hrank
    """
    try:
        if np.isnan(feature_map).any() or np.isinf(feature_map).any():
            print("Warning: NaN or Inf detected in feature map!")
            return 0  # 直接返回 0，避免 SVD 失败
        if len(feature_map.shape) == 1:  # 处理 1D BN 数据
            feature_map = feature_map.reshape(1, -1)
        else:
            H, W = feature_map.shape
            feature_map = feature_map.reshape(H, -1)

        U, S, Vh = np.linalg.svd(feature_map, full_matrices=False)
        rank = np.sum(S > 1e-10)
        return rank
    except Exception as e:
        return 0
def train_pruning_Hrank_pillar(model, optimizer, train_loader, train_set, MODEL, cfg, args, optim_cfg, start_epoch,
                        total_epochs, ckpt_save_dir, PRUNING2D, Random):
    """
      Hrank
    """
    dataloader_iter = iter(train_loader)

    features_dict = {}

    def hook_fn(module, input, output):
        if module not in features_dict:
            features_dict[module] = []
        if isinstance(module, (spconv.SparseConv3d, spconv.SubMConv3d)):
            features_dict[module].append(output.features.detach().cpu())
        else:
            features_dict[module].append(output.detach().cpu())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):  #pillar
            hooks.append(m.register_forward_hook(hook_fn))

    for _ in range(100):
        batch = next(dataloader_iter)
        load_data_to_gpu(batch)
        with torch.no_grad():
            model(batch)

    for hook in hooks:
        hook.remove()

    print("Computing Hrank-based pruning...")
    total = 0
    pruned = 0
    cfg_layer = []
    cfg_mask = []

    for k, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):  #pillar
            weight_copy = m.weight.data.clone()
            size = weight_copy.shape[0]
            total += size

            ranks = np.zeros(size)
            for f in features_dict[m]:
                if f is not None:
                    f_numpy = f.cpu().numpy()
                    for i in range(size):
                        if isinstance(m, (torch.nn.BatchNorm1d, spconv.SparseConv3d, spconv.SubMConv3d)):
                            data = f_numpy[:, i]
                            continue
                            # ranks[i] += np.var(data)
                            # ranks[i] += compute_rank(matrix_data)  # 累加 Rank
                        else:
                            ranks[i] += compute_rank(f_numpy[0, i, :, :])  # 累加 Rank

            thre_index = int(size * (1 - cfg.PRUNING3D.retain_ratio3D)) #remove - sto
            rank_sorted = np.sort(ranks)
            if Random:
                weight_copy = torch.rand_like(weight_copy)
                rank_sorted = np.sort(weight_copy.abs())
            thre = rank_sorted[thre_index]
            initial_mask = torch.tensor((ranks > thre), dtype=torch.float32).cuda()
            middle_mask = torch.tensor((ranks == thre), dtype=torch.float32).cuda()
            weights_sorted, _ = torch.sort(weight_copy.abs())
            thre_weight = weights_sorted[thre_index]
            weight_mask = weight_copy.abs().ge(thre_weight).float().cuda()
            middle_mask *= weight_mask
            mask = torch.add(initial_mask, middle_mask)
            num_to_remove = int(torch.sum(mask)) + thre_index - size
            if num_to_remove > 0:
                one_indices = torch.nonzero(mask).squeeze()
                if len(one_indices) > num_to_remove:
                    remove_indices = one_indices[torch.randperm(len(one_indices))[:num_to_remove]]
                    mask[remove_indices] = 0

            pruned += mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_layer.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print(f'Layer {k}: Total={size}, Remaining={int(torch.sum(mask))}')

    pruned_ratio = pruned / total
    print(f'Pruned Ratio: {pruned_ratio:.4f}')

    try:
        MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] = int(MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] * cfg.PRUNING3D.retain_ratio)
        MODEL['BACKBONE_3D']['WIDTH'] = MODEL['BACKBONE_3D']['WIDTH'] * cfg.PRUNING3D.retain_ratio
    except:
        print('MODEL[DBACKBONE_3D] is not exist.')
    if PRUNING2D:
        MODEL['BACKBONE_2D']['NUM_FILTERS'] = [int(filter_num * cfg.PRUNING3D.retain_ratio) for filter_num in
                                               MODEL['BACKBONE_2D']['NUM_FILTERS']]
        MODEL['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS'] = [int(filter_num * cfg.PRUNING3D.retain_ratio) for filter_num in
                                                        MODEL['BACKBONE_2D']['NUM_UPSAMPLE_FILTERS']]

        try:
            MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] = int(MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] * cfg.PRUNING3D.retain_ratio)
        except:
            print('MODEL[DENSE_HEAD][SHARED_CONV_CHANNEL] is not exist.')

    newmodel = build_network(model_cfg=MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    newmodel.cuda()

    layer_id_in_cfg = 0
    start_mask = np.ones(5)
    start_mask2D = np.ones(64)
    end_mask = cfg_mask[layer_id_in_cfg].cpu().numpy()
    for m0, m1 in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, torch.nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):
                start_mask2D = end_mask
                end_mask = cfg_mask[layer_id_in_cfg].cpu().numpy()
        elif isinstance(m0, (spconv.SparseConv3d, spconv.SubMConv3d)):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            if len(idx0) > m1.weight.data.shape[1]:
                idx0 = idx0[:m1.weight.data.shape[1]]

            if m0.weight.data.shape[-1] < len(idx0):
                w = m0.weight.data.clone()
            else:
                w = m0.weight.data[:, :, :, :, idx0].clone()
            w = w[idx1, :, :, :, :].clone()
            m1.weight.data = w.clone()

        elif isinstance(m0, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            if m0.out_channels in {1, 2, 3}:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask2D)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            if len(idx0) > m1.weight.data.shape[1]:
                idx0 = idx0[:m1.weight.data.shape[1]]
            if isinstance(m0,  torch.nn.ConvTranspose2d):
                idx_size_0 = m0.weight.data.size(1)
                idx0 = idx0[idx0 < idx_size_0]
                idx_size_1 = m1.weight.data.size(0)
                idx1 = idx1[idx1 < idx_size_1]
            for index in idx0:
                assert 0 <= index < m0.weight.data.size(1), f"idx0{idx0} out of bounds m0{m0.weight.data.shape}"

            if m0.weight.data.shape[1] <= len(idx0):
                w = m0.weight.data.clone()
            else:
                w = m0.weight.data[:, idx0, :, :].clone()
            for index in idx1:
                assert 0 <= index < w.size(0), f"idx1 {idx1} is out of bounds for dimension with size m0{w.shape} m1{m1.weight.data.shape}"
            m1.weight.data = w[idx1, :, :, :].clone()
        else:
            m1 = m0

    ckpt_name = ckpt_save_dir / (f'checkpoint_epoch_{start_epoch}')
    save_checkpoint(checkpoint_state(newmodel, optimizer, start_epoch), filename=ckpt_name)
    print(newmodel)
    print("Hrank-based pruning completed!")
    return newmodel

def train_pruning_Hrank_voxel(model, optimizer, train_loader, train_set, MODEL, cfg, args, optim_cfg, start_epoch,
                        total_epochs, ckpt_save_dir, PRUNING2D, Random):
    """
    Hrank voxel  +weights
    """
    dataloader_iter = iter(train_loader)

    features_dict = {}

    def hook_fn(module, input, output):
        if module not in features_dict:
            features_dict[module] = []
        if isinstance(module, (spconv.SparseConv3d, spconv.SubMConv3d)):
            features_dict[module].append(output.features.detach().cpu())
        else:
            features_dict[module].append(output.detach().cpu())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): #voxel
            hooks.append(m.register_forward_hook(hook_fn))

    for _ in range(100):
        batch = next(dataloader_iter)
        load_data_to_gpu(batch)
        with torch.no_grad():
            model(batch)

    for hook in hooks:
        hook.remove()

    print("Computing Hrank-based pruning...")
    total = 0
    pruned = 0
    cfg_layer = []
    cfg_mask = []

    for k, m in enumerate(model.modules()):
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            weight_copy = m.weight.data.clone()
            size = weight_copy.shape[0]
            total += size

            ranks = np.zeros(size)
            for f in features_dict[m]:
                if f is not None:
                    f_numpy = f.cpu().numpy()
                    for i in range(size):
                        if isinstance(m, (torch.nn.BatchNorm1d, spconv.SparseConv3d, spconv.SubMConv3d)):
                            data = f_numpy[:, i]
                            # ranks[i] += len(np.unique(data.astype(int)))  # number
                            ranks[i] += np.var(data)
                            # ranks[i] += compute_rank(matrix_data)
                        else:
                            ranks[i] += compute_rank(f_numpy[0, i, :, :])

            thre_index = int(size * (1 - cfg.PRUNING3D.retain_ratio3D)) #remove - sto
            rank_sorted = np.sort(ranks)
            if Random:
                weight_copy = torch.rand_like(weight_copy)
                rank_sorted = np.sort(weight_copy.abs())
            thre = rank_sorted[thre_index]
            initial_mask = torch.tensor((ranks > thre), dtype=torch.float32).cuda()
            middle_mask = torch.tensor((ranks == thre), dtype=torch.float32).cuda()
            weights_sorted, _ = torch.sort(weight_copy.abs())
            thre_weight = weights_sorted[thre_index]
            weight_mask = weight_copy.abs().ge(thre_weight).float().cuda()
            middle_mask *= weight_mask
            mask = torch.add(initial_mask, middle_mask)
            num_to_remove = int(torch.sum(mask)) + thre_index - size
            if num_to_remove > 0:
                one_indices = torch.nonzero(mask).squeeze()
                if len(one_indices) > num_to_remove:
                    remove_indices = one_indices[torch.randperm(len(one_indices))[:num_to_remove]]
                    mask[remove_indices] = 0

            pruned += mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_layer.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print(f'Layer {k}: Total={size}, Remaining={int(torch.sum(mask))}')

    pruned_ratio = pruned / total
    print(f'Pruned Ratio: {pruned_ratio:.4f}')

    try:
        MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] = int(MODEL['MAP_TO_BEV']['NUM_BEV_FEATURES'] * cfg.PRUNING3D.retain_ratio)
        MODEL['BACKBONE_3D']['WIDTH'] = MODEL['BACKBONE_3D']['WIDTH'] * cfg.PRUNING3D.retain_ratio
    except:
        print('MODEL[DBACKBONE_3D] is not exist.')
    if PRUNING2D:
        MODEL['BACKBONE_2D']['WIDTH'] = MODEL['BACKBONE_2D']['WIDTH'] * cfg.PRUNING3D.retain_ratio3D
        try:
            MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] = int(MODEL['DENSE_HEAD']['SHARED_CONV_CHANNEL'] * cfg.PRUNING3D.retain_ratio)
        except:
            print('MODEL[DENSE_HEAD][SHARED_CONV_CHANNEL] is not exist.')

    newmodel = build_network(model_cfg=MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    newmodel.cuda()

    layer_id_in_cfg = 0
    start_mask = np.ones(5)
    start_mask2D = np.ones(64)
    end_mask = cfg_mask[layer_id_in_cfg].cpu().numpy()
    for m0, m1 in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, (torch.nn.BatchNorm1d)):  #voxel
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):
                start_mask2D = np.concatenate((start_mask, end_mask))
                start_mask = end_mask
                end_mask = cfg_mask[layer_id_in_cfg].cpu().numpy()

        elif isinstance(m0, torch.nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            m1.weight.data = m0.weight.data[idx1].clone()
            m1.bias.data = m0.bias.data[idx1].clone()
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):
                start_mask2D = end_mask
                end_mask = cfg_mask[layer_id_in_cfg].cpu().numpy()
        elif isinstance(m0, (spconv.SparseConv3d, spconv.SubMConv3d)):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            if len(idx0) > m1.weight.data.shape[1]:
                idx0 = idx0[:m1.weight.data.shape[1]]

            if m0.weight.data.shape[-1] < len(idx0):
                w = m0.weight.data.clone()
            else:
                w = m0.weight.data[:, :, :, :, idx0].clone()
            w = w[idx1, :, :, :, :].clone()
            m1.weight.data = w.clone()

        elif isinstance(m0, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            if m0.out_channels in {1, 2, 3}:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask2D)))
            if len(idx1) > m1.weight.data.shape[0]:
                idx1 = idx1[:m1.weight.data.shape[0]]
            if len(idx0) > m1.weight.data.shape[1]:
                idx0 = idx0[:m1.weight.data.shape[1]]
            if isinstance(m0,  torch.nn.ConvTranspose2d):
                idx_size_0 = m0.weight.data.size(1)
                idx0 = idx0[idx0 < idx_size_0]
                idx_size_1 = m1.weight.data.size(0)
                idx1 = idx1[idx1 < idx_size_1]
            for index in idx0:
                assert 0 <= index < m0.weight.data.size(1), f"idx0{idx0} out of bounds m0{m0.weight.data.shape}"

            if m0.weight.data.shape[1] <= len(idx0):
                w = m0.weight.data.clone()
            else:
                w = m0.weight.data[:, idx0, :, :].clone()
            for index in idx1:
                assert 0 <= index < w.size(0), f"idx1 {idx1} is out of bounds for dimension with size m0{w.shape} m1{m1.weight.data.shape}"
            m1.weight.data = w[idx1, :, :, :].clone()
        else:
            m1 = m0

    ckpt_name = ckpt_save_dir / (f'checkpoint_epoch_{start_epoch}')
    save_checkpoint(checkpoint_state(newmodel, optimizer, start_epoch), filename=ckpt_name)
    print(newmodel)
    print("Hrank-based pruning completed!")
    return newmodel

def global_prune(model, prune_ratio, optimizer, start_epoch, ckpt_save_dir, only_3d=True, only_2d=False):
    import logging
    logging.basicConfig(filename='global_prune.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info('\n\n################')
    logging.info(f'global: only_3d= {only_3d}, prune_ratio {prune_ratio}.')
    logging.info(f'ckpt_save_dir/epoch.pth: {ckpt_save_dir}/checkpoint_epoch_{start_epoch}')

    all_weights = []
    if only_3d:
        for name, module in model.named_modules():
            if isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SubMConv3d):
                all_weights.append(module.weight.data.view(-1))
    elif only_2d:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.out_channels not in (1, 2, 3):
                all_weights.append(module.weight.data.view(-1))
            elif isinstance(module, torch.nn.ConvTranspose2d):
                all_weights.append(module.weight.data.view(-1))
    else:
        for name, module in model.named_modules():
            if isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SubMConv3d):
                all_weights.append(module.weight.data.view(-1))

            elif isinstance(module, torch.nn.Conv2d) and module.out_channels not in (1,2,3):
                all_weights.append(module.weight.data.view(-1))
            elif isinstance(module, torch.nn.ConvTranspose2d):
                all_weights.append(module.weight.data.view(-1))

    all_weights = torch.cat(all_weights)
    all_importance = torch.abs(all_weights)

    y, i = torch.sort(all_importance)
    thres_index = int(len(y) * prune_ratio)
    threshold = y[thres_index]


    ratio_M = []
    for name, module in model.named_modules():
        if only_3d and not only_2d:
            if isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SubMConv3d):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned / mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned / mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
        elif only_2d and not only_3d:
            if isinstance(module, torch.nn.Conv2d) and module.out_channels not in (1, 2, 3):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned / mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned / mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
            elif isinstance(module, torch.nn.ConvTranspose2d):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned / mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned / mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
        else:
            if isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SubMConv3d):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned/mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned/mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
            elif isinstance(module, torch.nn.Conv2d) and module.out_channels not in (1,2,3):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)  #大于等于阈值的保留
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned/mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned/mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
            elif isinstance(module, torch.nn.ConvTranspose2d):
                weight_importance = torch.abs(module.weight.data)
                mask = weight_importance.ge(threshold)  #大于等于阈值的保留
                pruned = mask.numel() - mask.sum().item()
                ratio = "{:.2f}".format(pruned/mask.numel() * 100)
                ratio_M.append("{:.4f}".format(pruned/mask.numel()))
                module.weight.data *= mask
                logging.info(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio}%')
                print(f'name {name}: Globally pruned {pruned} weights, total {mask.numel()} {ratio} %')
    logging.info(f'ratio_M {ratio_M}')
    print(f'ratio_M {ratio_M}')
    # ------------------- save checkpoint ------------------------------------------------------
    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % start_epoch)
    save_checkpoint(
        checkpoint_state(model, optimizer, start_epoch), filename=ckpt_name,
    )
    print("Build new model successfully")
    print(model)
    return model

def global_unstructued_prune(model, optimizer, train_loader, train_set, MODEL, cfg, args, optim_cfg, start_epoch, total_epochs, ckpt_save_dir, PRUNING2D):
    import torch.nn.utils.prune as prune
    import logging
    logging.basicConfig(filename='Global_sparsity.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    pru_amount = 0.1
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pru_amount)  #amount代表去掉的比例
        elif isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=pru_amount)

        if isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SubMConv3d):
            prune.l1_unstructured(module, name='weight', amount=pru_amount)
        elif isinstance(module, torch.nn.BatchNorm1d):
            prune.l1_unstructured(module, name='weight', amount=pru_amount)
        else:
            continue

        logging.info("Global sparsity: {:.2f}%".format(100. * float(torch.sum(module.weight == 0))
                                               / float(module.weight.nelement())))

    print(dict(model.named_buffers()).keys())
    print(model)
    # ------------------- save checkpoint ------------------------------------------------------
    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % start_epoch)
    save_checkpoint(
        checkpoint_state(model, optimizer, start_epoch), filename=ckpt_name,
    )
    return model

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
