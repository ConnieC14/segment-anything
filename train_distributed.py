# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import argparse
import random
from datetime import datetime
import shutil
import monai

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from segment_anything import sam_model_registry

from semisupSAM.dataset import BraTS2021Dataset
from semisupSAM.plot import (show_mask, show_box,
                             plot_comparison, plot_loss)
from semisupSAM.utils import (EarlyStopper, get_path_files,
                              convert_outputs_to_masks,
                              compute_pixel_accuracy, 
                              compute_precision_recall_f1,
                              iou_pytorch,
                              save_test_data)
from semisupSAM.model import MRISAM

SEED = 23

# set seeds
torch.manual_seed(SEED)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()


def main_worker(gpu, ngpus_per_node, args):
    # %% set up model for training
    # device = args.device
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)

    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(
            __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
        )
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl", init_method=args.init_method, rank=rank, world_size=world_size
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MRISAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system("nvidia-smi")

    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[1] / (
        1024**3
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb"
    )
    print(
        f"[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb"
    )
    if rank % ngpus_per_node == 0:
        print("After DDP initialization:")
        os.system("nvidia-smi")

    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    ## Setting up optimiser and loss func
    # only optimize the parameters of image encodder, mask decoder, do not update prompt encoder
    # img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    img_mask_encdec_params = list(
        medsam_model.module.image_encoder.parameters()
    ) + list(medsam_model.module.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    mIoUs = []
    pred_mIoUs = []

    best_loss = 1e10

    # add in data information 
    img_path_files = get_path_files(args.tr_data_path, args.mri_scan)
    train, test, val = np.split(img_path_files, [int(len(img_path_files)*0.7), int(len(img_path_files)*0.85)])

    train_dataset = BraTS2021Dataset(args.tr_data_path, train)
    test_dataset = BraTS2021Dataset(args.tr_data_path, test)
    val_dataset = BraTS2021Dataset(args.tr_data_path, val)
    
    if args.use_wandb:
        # save test files
        test_folder_name = 'test_dataset/'
        test_output_name = join(args.work_dir, test_folder_name)
        test_output_name = join(test_output_name, args.task_name + "-" + run_id + '.pth')
        save_test_data(test_dataset, test_folder_name, test_output_name)

    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    print('Total images: ', len(img_path_files))
    print('Training: ', len(train_dataset))
    print('Validation: ', len(val_dataset))
    print('Testing: ', len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                rank,
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    for epoch in range(start_epoch, num_epochs):
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        epoch_loss = 0
        mIoU = 0
        pred_mIoU = 0

        train_dataloader.sampler.set_epoch(epoch)
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]")):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            
            image, gt2D = image.cuda(), gt2D.cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred, iou_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred, iou_pred = medsam_model(image, boxes_np)
                iou = iou_pytorch(medsam_pred.detach().cpu(), gt2D.detach().cpu())
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                    medsam_pred, gt2D.float()
                )
                
                # Gradient accumulationgt
                if args.grad_acc_steps > 1:
                    loss = (
                        loss / args.grad_acc_steps
                    )  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with medsam_model.no_sync():
                            loss.backward()  ## calculate the gradient only
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            if step > 10 and step % 100 == 0:
                if is_main_host:
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint,
                        join(model_save_path, "medsam_model_latest_step.pth"),
                    )

            mIoU += iou
            pred_mIoU += np.mean(iou_pred.detach().cpu().numpy())
            epoch_loss += loss.item()
            iter_num += 1

        # Check CUDA memory usage
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0] / (1024**3), cuda_mem_info[
            1
        ] / (1024**3)
        print("\n")
        print(f"[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb")
        print(f"[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb")
        print(
            f"[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb"
        )
        print("\n")

        # Save and log train loss/metrics & information
        epoch_loss /= step
        mIoU /= step
        pred_mIoU /= step

        # Save metrics
        losses.append(epoch_loss)
        mIoUs.append(mIoU)
        pred_mIoUs.append(pred_mIoU)

        if args.use_wandb:
            args.run.log({"train_epoch_loss": epoch_loss})
            args.run.log({"train_mIoU": mIoU})
            args.run.log({"train_pred_mIoU": pred_mIoU})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        # save the model checkpoint
        if is_main_host:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))

            ## save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% Validation after each epoch
        medsam_model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            val_loss = 0.0
            val_mIou = 0.0
            val_pred_mIou = 0.0
            val_px_acc = 0.0
            val_prec, val_recall, val_f1 = 0.0, 0.0, 0.0
            batch_size = len(val_dataloader)

            for step, (image, mask2D, boxes, _) in enumerate(tqdm(val_dataloader)):
                boxes_np = boxes.detach().cpu().numpy()
                image, mask2D = image.cuda(), mask2D.cuda()

                medsam_pred, iou_pred = medsam_model(image, boxes_np)

                # Adjust predicted mask to be binary
                pred_mask = convert_outputs_to_masks(medsam_pred, threshold=0.5)
                
                # Save IoU metric
                val_pred_mIou += np.mean(iou_pred.detach().cpu().numpy())
                val_mIou += iou_pytorch(medsam_pred.detach().cpu(), mask2D.detach().cpu())

                val_loss += seg_loss(medsam_pred, mask2D) + ce_loss(medsam_pred, mask2D.float())
                
                # Early stop if validation loss is not improving
                if early_stopper.early_stop(val_loss):
                    break
                
                # Calculate pixel accuracy, prec, recall, f1 for the batch
                val_px_acc += compute_pixel_accuracy(pred_mask, mask2D.detach().cpu())
                val_p, val_r, val_f = compute_precision_recall_f1(pred_mask, mask2D.detach().cpu())
                val_prec += val_p
                val_recall += val_r
                val_f1 += val_f

            # Print validation statistics
            val_loss /= batch_size
            val_mIou /= batch_size
            val_pred_mIou /= batch_size
            val_prec /= batch_size
            val_recall /= batch_size
            val_f1 /= batch_size
            val_px_acc /= batch_size

            print(f"Validation Loss: {val_loss:.4f}, \
                    Validation Pixel Accuracy: {val_px_acc:.2f}%, \
                    Validation mIoU: {val_mIou:.2f}")
            
            if args.use_wandb:
                args.run.log({"val_epoch_loss": val_loss})
                args.run.log({"val_mIoU": mIoU})
                args.run.log({"val_pred_mIou": val_pred_mIou})
                args.run.log({"val_precision": val_prec})
                args.run.log({"val_recall": val_recall})
                args.run.log({"val_f1": val_f1})
                args.run.log({"val_px_acc": val_px_acc})
        
        
        torch.distributed.barrier()
        
        # %% plot loss
        plot_loss(losses, join(model_save_path, args.task_name + "train_loss.png"))
        
        if epoch % 10 == 0:
            fig = plot_comparison(image, boxes, mask2D, pred_mask)
            fig.savefig(join(model_save_path, args.task_name + f"val_n_epoch_{epoch}_int_mask.png"))
            plt.close(fig)

            fig = plot_comparison(image, boxes, mask2D, medsam_pred)
            fig.savefig(join(model_save_path, args.task_name + f"val_n_epoch_{epoch}_float_mask.png"))
            plt.close(fig)

    args.run.finish()
    

def main():

    # %% sanity test of dataset class
    mri_scan = "T2"

    img_path_files = get_path_files(args.tr_data_path, mri_scan)

    tr_dataset = BraTS2021Dataset("./data/BraTS2021/BraTS2021Training_Data", files=img_path_files) # TODO: Check this path
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, (image, mask, bboxes, scan_name) in enumerate(tr_dataloader):
        print(image.shape, mask.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        idx = random.randint(0, 7)
        axs[0].imshow(image[idx][[0],:,:].cpu().permute(1, 2, 0).numpy())
        show_mask(mask[idx].cpu().numpy(), axs[0])
        if (bboxes.sum() > 0):
            show_box(bboxes[idx].numpy(), axs[0])
        axs[0].axis("off")
        # set title
        axs[0].set_title(scan_name[idx])
        idx = random.randint(0, 7)
        axs[1].imshow(image[idx][[0],:,:].cpu().permute(1, 2, 0).numpy())
        show_mask(mask[idx].cpu().numpy(), axs[1])
        if (bboxes.sum() > 0):
            show_box(bboxes[idx].numpy(), axs[1])
        axs[1].axis("off")
        # set title
        axs[1].set_title(scan_name[idx])
        # plt.show()
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
        plt.close()
        break
    
    print("Spawning processces")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(main_worker, nprocs=args.num_of_gpus, args=(args.num_of_gpus, args))


if __name__ == "__main__":
    # %% set up parser
    parser.add_argument(
        "-i", "--tr_data_path",
        type=str, default="./data/BraTS2021/BraTS2021Training_Data",
        help="path to training files; two subfolders: gts and imgs",
    )
    parser.add_argument("-mri_scan", type=str, default="T2")
    parser.add_argument("-task_name", type=str, default="MRI_SAM")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--load_pretrain", type=bool, default=True, help="use wandb to monitor training")
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default=os.getcwd())
    # train
    parser.add_argument("-num_epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=6)
    parser.add_argument("-num_workers", type=int, default=6)
    # Optimizer parameters
    parser.add_argument( "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "-use_wandb", type=bool, default=True, help="use wandb to monitor training" # TODO: FIX BACK TO FALSE
    )
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    ## Distributed training args
    parser.add_argument("--world_size", type=int, help="world size")
    parser.add_argument("--node_rank", type=int, default=0, help="Node rank")
    parser.add_argument(
        "--bucket_cap_mb", type=int, default=25,
        help="The amount of memory in Mb that DDP will accumulate before \
        firing off gradient communication for the bucket (need to tune)",
    )
    parser.add_argument(
        "--grad_acc_steps", type=int, default=1,
        help="Gradient accumulation steps before syncing gradients for backprop",
    )
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--init_method", type=str, default="env://")

    args = parser.parse_args()

    if args.use_wandb: 
        import wandb
        
        args.run = wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_data_path,
                "model_type": args.model_type,
            },
            group="DDP"
        )

    if not args.world_size:
        args.num_of_gpus = torch.cuda.device_count()
        num_processes = 1
        num_nodes = 1
        args.world_size = args.num_of_gpus*num_nodes*num_processes
    
    main()

