# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back, flip_back_simdr
from utils.transforms import transform_preds
from utils.vis import save_debug_images
from core.loss import JointsMSELoss, NMTCritierion


logger = logging.getLogger(__name__)

def train_sa_simdr(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_x, target_y, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target_x, target_y, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_sa_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target_x, target_y, target_weight, meta) in enumerate(val_loader):
            # compute output
            output_x, output_y = model(input)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                output_x_flipped_, output_y_flipped_ = model(input_flipped)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]                                                         
                output_x = F.softmax((output_x+output_x_flipped)*0.5,dim=2)
                output_y = F.softmax((output_y+output_y_flipped)*0.5,dim=2)
            else:
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)                                


            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            loss = criterion(output_x, output_y, target_x, target_y, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)
            
            mask = max_val_x > max_val_y
            max_val_x[mask] = max_val_y[mask]
            maxvals = max_val_x.cpu().numpy()

            output = torch.ones([input.size(0),preds_x.size(1),2])
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            output = output.cpu().numpy()
            preds = output.copy()
            # Transform back
            for j in range(output.shape[0]):
                preds[j] = transform_preds(
                    output[j], c[j], s[j], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
                )

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, None, preds, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def train_simdr(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_x, output_y = model(input)

        target = target.cuda(non_blocking=True).long()
        target_weight = target_weight.cuda(non_blocking=True).float()


        loss = criterion(output_x, output_y, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_simdr(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    # 记录批处理时间和损失
    batch_time = AverageMeter()
    losses = AverageMeter()

    # 切换到评估模式
    model.eval()

    # 初始化用于存储所有预测结果和 box 信息 numpy 数组
    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3), # 存储每个样本、每个关节点的 (x, y, confidence)
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6)) # 存储 box 信息，通常是 (x1, y1, x2, y2, score, etc.)
    image_path = [] # 存储图像路径
    filenames = [] # 存储文件名
    imgnums = [] # 存储图像编号
    idx = 0 # 记录当前处理到的样本索引

    # 在 no_grad 模式下进行推理，不计算梯度
    with torch.no_grad():
        end = time.time()
        # 遍历验证集加载器
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # model(input) 返回 (output_x, output_y)，形状均为 [b, num_keypoints, logits_dim]
            output_x, output_y = model(input)

            # 处理翻转测试 (Flip Test)
            if config.TEST.FLIP_TEST:
                # 对输入图像进行水平翻转
                input_flipped = input.flip(3)
                # 对翻转后的图像进行模型前向传播
                output_x_flipped_, output_y_flipped_ = model(input_flipped)

                # 对翻转图像的预测结果进行反翻转
                # flip_back_simdr 函数用于将翻转图像上的预测结果转换回原始图像坐标系
                # type='x' 表示对 x 坐标进行反翻转，type='y' 表示对 y 坐标进行反翻转 (通常 y 坐标不需要反翻转)
                output_x_flipped = flip_back_simdr(output_x_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='x')
                output_y_flipped = flip_back_simdr(output_y_flipped_.cpu().numpy(),
                                           val_dataset.flip_pairs,type='y')

                # 将 numpy 数组转换回 PyTorch 张量并移到 GPU
                output_x_flipped = torch.from_numpy(output_x_flipped.copy()).cuda()
                output_y_flipped = torch.from_numpy(output_y_flipped.copy()).cuda()

                # 可选：对翻转热图进行微小偏移，提高精度 (针对某些不对称的热图)
                if config.TEST.SHIFT_HEATMAP:
                    output_x_flipped[:, :, 0:-1] = \
                        output_x_flipped.clone()[:, :, 1:]

                # 将原始预测结果和反翻转后的预测结果进行平均 (在概率空间进行平均)
                # F.softmax 将 logits 转换为概率分布
                output_x = (F.softmax(output_x,dim=2) + F.softmax(output_x_flipped,dim=2))*0.5
                output_y = (F.softmax(output_y,dim=2) + F.softmax(output_y_flipped,dim=2))*0.5
            else:
                # 不进行翻转测试时，直接将 logits 转换为概率分布
                output_x = F.softmax(output_x,dim=2)
                output_y = F.softmax(output_y,dim=2)

            # 将真实标签和权重移到 GPU
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True).float()

            # 计算损失
            # 注意：这里计算损失时用的是 Softmax 后的概率分布作为输入，而不是 LogSoftmax 后的对数概率
            # criterion 的实现内部会再做一次 LogSoftmax
            # 这可能是一个实现上的细节，或者 criterion 被设计成可以处理概率输入
            loss = criterion(output_x, output_y, target, target_weight)

            # 记录损失和样本数量
            num_images = input.size(0)
            losses.update(loss.item(), num_images)

            # 记录批处理时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 获取 meta 信息中的中心点、尺度和分数
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            # --- SimCC 坐标解码部分 ---
            # 从预测的概率分布中找到最大概率对应的索引
            # output_x 的形状是 (b, num_keypoints, x_logits_dim)
            # output_y 的形状是 (b, num_keypoints, y_logits_dim)
            # max(2, keepdim=True) 在最后一个维度（logits 维度）找到最大值和对应的索引
            # max_val_x, max_val_y 的形状是 (b, num_keypoints, 1)
            # preds_x, preds_y 的形状是 (b, num_keypoints, 1)，存储的是离散化后的索引
            max_val_x, preds_x = output_x.max(2,keepdim=True)
            max_val_y, preds_y = output_y.max(2,keepdim=True)

            # 确定预测位置的置信度：
            # 这里使用了一个策略，取 x 和 y 预测中概率较大的那个作为关节点的置信度
            # mask = max_val_x < max_val_y
            # max_val_x[mask] = max_val_y[mask]
            # 或者简单平均 (代码中注释掉了)
            # max_val_x = (max_val_x + max_val_y)/2
            maxvals = max_val_x.cpu().numpy() # 将置信度转换为 numpy 数组

            # 将预测的离散索引转换回原始热图尺度下的连续坐标
            # output 初始化为 (b, num_keypoints, 2)
            output = torch.ones([input.size(0),preds_x.size(1),2])
            # preds_x 和 preds_y 存储的是在离散化维度上的索引 (0, 1, ..., N-1)
            # config.MODEL.SIMDR_SPLIT_RATIO 是离散化的步长因子
            # 索引除以 SIMDR_SPLIT_RATIO 将索引转换回在原热图尺寸 (HEATMAP_SIZE) 下的坐标值
            # 例如，如果热图尺寸是 64，SIMDR_SPLIT_RATIO 是 2，那么 logits_dim 可能是 128
            # 索引 64 对应于热图尺寸下的 64 / 2 = 32
            output[:,:,0] = torch.squeeze(torch.true_divide(preds_x, config.MODEL.SIMDR_SPLIT_RATIO))
            output[:,:,1] = torch.squeeze(torch.true_divide(preds_y, config.MODEL.SIMDR_SPLIT_RATIO))

            # 将预测的坐标 (现在是热图尺度下的浮点坐标) 转换为 numpy 数组
            output = output.cpu().numpy()
            preds = output.copy()
            # --- SimCC 坐标解码部分结束 ---

            # 将热图尺度下的预测坐标转换回原始图像尺度下的坐标
            # transform_preds 是一个外部函数，负责将基于热图的坐标通过中心点和尺度信息转换回原始图像坐标
            for j in range(output.shape[0]):
                preds[j] = transform_preds(
                    output[j], c[j], s[j], [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]] # 使用原始图像尺寸进行转换
                )

            # 存储当前批次的预测结果、置信度和 box 信息
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2] # 存储转换后的 x, y 坐标
            all_preds[idx:idx + num_images, :, 2:3] = maxvals # 存储置信度
            # 存储 box 信息 (中心点, 尺度, 面积, 分数)
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1) # 根据尺度估算 box 面积
            all_boxes[idx:idx + num_images, 5] = score # 存储检测框分数

            # 存储图像路径
            image_path.extend(meta['image'])

            # 更新样本索引
            idx += num_images

            # 打印测试进度和损失
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

                # 保存调试图像（可选）
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images 函数用于保存中间结果的可视化图像
                save_debug_images(config, input, meta, target, preds, output,
                                  prefix)

        # 调用评估工具计算性能指标 (例如 AP)
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        # 打印评估结果
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        # 将评估结果写入 TensorBoard (可选)
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    # 返回性能指标
    return perf_indicator

def train_heatmap(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate_heatmap(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
