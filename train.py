# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import wandb
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
import torch.nn.functional as F

from segment_anything import sam_model_registry
from semisupSAM.dataset import BraTS2021Dataset
from semisupSAM.utils import (EarlyStopper, get_path_files,
                              convert_outputs_to_masks,
                              compute_pixel_accuracy, 
                              compute_precision_recall_f1,
                              iou_pytorch,
                              save_test_data)
from semisupSAM.plot import (show_mask, show_box,
                             plot_comparison, plot_loss)
from semisupSAM.model import MRISAM

SEED = 23

# set seeds
torch.manual_seed(SEED)
torch.cuda.empty_cache()


def main():
    #  Get full list of directory files
    img_path_files = get_path_files(args.tr_data_path, args.mri_scan)

    # %% sanity test of dataset class
    tr_dataset = BraTS2021Dataset(args.tr_data_path, files=img_path_files)
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    for step, (image, mask, bboxes, scan_name) in enumerate(tr_dataloader):
        print(image.shape, mask.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        idx = random.randint(0, args.batch_size-1)
        axs[0].imshow(image[idx][[0],:,:].cpu().permute(1, 2, 0).numpy())
        show_mask(mask[idx].cpu().numpy(), axs[0])
        if (bboxes.sum() > 0):
            show_box(bboxes[idx].numpy(), axs[0])
        axs[0].axis("off")
        # set title
        axs[0].set_title(scan_name[idx])
        idx = random.randint(0, args.batch_size-1)
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
    
    # %% set up model for training
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    
    # %% set up model
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MRISAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    mIoUs = []
    pred_mIoUs = []

    best_loss = 1e10

    # %% Read image fiels
    img_path_files = get_path_files(args.tr_data_path, args.mri_scan)

    train, test, val = np.split(img_path_files, [int(len(img_path_files)*0.7), int(len(img_path_files)*0.85)])

    train_dataset = BraTS2021Dataset(args.tr_data_path, train)
    test_dataset = BraTS2021Dataset(args.tr_data_path, test)
    val_dataset = BraTS2021Dataset(args.tr_data_path, val)

    if args.use_wandb:
        # save test files
        test_folder_name = 'test_dataset'
        test_output_name = join(args.work_dir, test_folder_name)
        test_output_name = join(test_output_name, args.task_name + "-" + run_id + '.pth')
        save_test_data(test_dataset, test_folder_name, test_output_name)

    # Print out dataset information
    print('Total images: ', len(img_path_files))
    print('Training: ', len(train_dataset))
    print('Validation: ', len(val_dataset))
    print('Testing: ', len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
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
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, num_epochs):
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        epoch_loss = 0
        mIoU = 0
        pred_mIoU = 0

        for step, (image, mask2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, mask2D = image.to(device), mask2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred, iou_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, mask2D) + ce_loss(medsam_pred, mask2D.float())
                    iou = iou_pytorch(medsam_pred.detach().cpu(), mask2D.detach().cpu())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred, iou_pred = medsam_model(image, boxes_np)
                iou = iou_pytorch(medsam_pred.detach().cpu(), mask2D.detach().cpu())
                loss = seg_loss(medsam_pred, mask2D) + ce_loss(medsam_pred, mask2D.float())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            
            mIoU += iou
            pred_mIoU += np.mean(iou_pred.detach().cpu().numpy())
            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        mIoU /= step
        pred_mIoU /= step

        # Save metrics
        losses.append(epoch_loss)
        mIoUs.append(mIoU)
        pred_mIoUs.append(pred_mIoU)

        if args.use_wandb:
            wandb.log({"train_epoch_loss": epoch_loss})
            wandb.log({"train_mIoU": mIoU})
            wandb.log({"train_pred_mIoU": pred_mIoU})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
        
        # Validation after each epoch
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
                image, mask2D = image.to(device), mask2D.to(device)

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
                    Validation mIoU: {val_mIou:.2f} \
                    Predicted Validation mIoU: {val_pred_mIou:.2f}")
            print(f"Precision: {val_prec:.4f}, \
                    Recall: {val_recall:.2f}%, \
                    F1 Score: {val_f1:.2f}")
            if args.use_wandb:
                wandb.log({"val_epoch_loss": val_loss})
                wandb.log({"val_mIoU": mIoU})
                wandb.log({"val_pred_mIou": val_pred_mIou})
                wandb.log({"val_precision": val_prec})
                wandb.log({"val_recall": val_recall})
                wandb.log({"val_f1": val_f1})
                wandb.log({"val_px_acc": val_px_acc})
        
        # %% plot loss
        plot_loss(losses, join(model_save_path, args.task_name + "train_loss.png"))

        fig = plot_comparison(image, boxes, mask2D, pred_mask)
        fig.savefig(join(model_save_path, args.task_name + f"val_n_epoch_{epoch}_int_mask.png"))
        plt.close(fig)

        fig = plot_comparison(image, boxes, mask2D, medsam_pred)
        fig.savefig(join(model_save_path, args.task_name + f"val_n_epoch_{epoch}_float_mask.png"))
        plt.close(fig)


if __name__ == "__main__":
    
    # %% set up parser
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-num_epochs", type=int, default=3)
    parser.add_argument("-batch_size", type=int, default=3)
    parser.add_argument("-num_workers", type=int, default=0)
    # Optimizer parameters
    parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
    ) # TODO: change use_wandb
    parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb

        # wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_data_path,
                "model_type": args.model_type,
            },
        )
    main()
