# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  Enhanced training script with early stopping and weight saving
# Author:       Sedigheh Eslami 
# Date:         2021/08/06
# Modified:     Added early stopping and model weight saving functionality
#-------------------------------------------------------------------------------

# Configuration for model saving
MODEL_SAVE_NAME = "Contrastive_loss"  # Change this name as needed
WEIGHTS_FOLDER = "./weights"    # Folder to save model weights
EARLY_STOPPING_PATIENCE = 5   # Number of epochs to wait before early stopping

import os
import time
import torch
from utils import utils
from datetime import datetime
import torch.nn as nn
from torch import optim
import clip
from utils.utils import (
        create_logger,
        get_optimizer,
        )
from core.function import valid_model
from core.evaluate import AverageMeter
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses


def _convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.best_weights = model.state_dict().copy()


# Train phase
def train(cfg, train_loader, eval_loader, device):
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir)
    logger, _ = create_logger(cfg)
    
    # Create directories
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create weights directory
    if not os.path.exists(WEIGHTS_FOLDER):
        os.makedirs(WEIGHTS_FOLDER)
    
    logger.info(f"-------Loading CLIP with vision encoder {cfg.TRAIN.VISION_ENCODER} -------")
    model, preprocess = clip.load(cfg.TRAIN.VISION_ENCODER, device=device, jit=False)
    if device == "cpu":
          model.float()
    else :
        clip.model.convert_weights(model)

    loss_img = losses.ContrastiveLoss()
    loss_txt = losses.ContrastiveLoss()
    optim = get_optimizer(cfg, model)

    best_loss, best_epoch, best_model = 10000, 0, ""
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, 
        restore_best_weights=True
    )

    logger.info("-------Training started-------")
    logger.info(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    logger.info(f"Model weights will be saved to: {WEIGHTS_FOLDER}")

    for epoch in range(cfg.TRAIN.N_EPOCH):
        train_all_loss = AverageMeter()
        model.train()
        model_save_path = os.path.join(
                model_dir,
                f"epoch_{epoch}.pth")
        number_batch = len(train_loader)

        # Predicting and computing score
        for i, (image, caption) in enumerate(train_loader):
            optim.zero_grad()
            images = torch.stack([img for img in image], dim=0).to(device)
            captions = clip.tokenize(caption, context_length=cfg.TRAIN.MAX_SEQ_LENGTH).to(device)


            logits_per_image, logits_per_text = model(images, captions)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            ground_truth = torch.arange(cfg.TRAIN.BATCH_SIZE, dtype=torch.long, device=device)
            lambdaa = 0.5
            train_total_loss = lambdaa*(loss_img(logits_per_image, ground_truth)) + (1-lambdaa)* (loss_txt(logits_per_text, ground_truth))
            train_total_loss.backward()
            if device == "cpu":
                optim.step()
            else : 
                _convert_models_to_fp32(model)
                optim.step()
                clip.model.convert_weights(model)
            if i % cfg.SHOW_STEP == 0:
                pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  ".format(epoch, i, number_batch, train_total_loss)
                logger.info(pbar_str)

            cnt = len(caption)
            train_all_loss.update(train_total_loss.data.item(), cnt)

        train_all_loss = train_all_loss.avg
        pbar_str = f"---Epoch:{epoch}  Epoch_Loss:{train_all_loss}"
        logger.info(pbar_str)
        writer.add_scalar("Loss/train", train_all_loss, epoch)

        # Eval
        if eval_loader is not None:
            eval_all_loss = valid_model(eval_loader, model, loss_img, loss_txt, cfg, device)
            
            # Check for early stopping
            if early_stopping(eval_all_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best validation loss: {early_stopping.best_loss}")
                break
            
            if eval_all_loss < best_loss:
                best_loss = eval_all_loss
                best_epoch = epoch
                best_model = model
                
                # Save best model in original location
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
                
                # Save model weights in weights folder
                weight_save_path = os.path.join(WEIGHTS_FOLDER, f"{MODEL_SAVE_NAME}_epoch_{epoch}_loss_{eval_all_loss:.4f}.pth")
                torch.save(model.state_dict(), weight_save_path)
                logger.info(f"Model weights saved to: {weight_save_path}")
                
            logger.info(
                    f"--------------Epoch:{epoch}    Eval_Loss:{eval_all_loss}%--------------")
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Eval_Loss:{best_loss}%--------------")
            writer.add_scalar("Loss/val", eval_all_loss, epoch)
        else:
            # Check for early stopping using training loss when no validation set
            if early_stopping(train_all_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best training loss: {early_stopping.best_loss}")
                break
                
            if train_all_loss < best_loss:
                best_loss = train_all_loss
                best_epoch = epoch
                best_model = model
                
                # Save best model in original location
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
                
                # Save model weights in weights folder
                weight_save_path = os.path.join(WEIGHTS_FOLDER, f"{MODEL_SAVE_NAME}_epoch_{epoch}_loss_{train_all_loss:.4f}.pth")
                torch.save(model.state_dict(), weight_save_path)
                logger.info(f"Model weights saved to: {weight_save_path}")
                
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Train_Loss:{best_loss}%--------------")
            writer.add_scalar("Loss/train-as-val", train_all_loss, epoch)

        if not os.path.exists(cfg.RESULTS_DIR):
            os.makedirs(cfg.RESULTS_DIR)

        with open(os.path.join(cfg.RESULTS_DIR, "best.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_loss": best_loss}, f)
    
    # Save final model weights
    final_weight_path = os.path.join(WEIGHTS_FOLDER, f"{MODEL_SAVE_NAME}_final.pth")
    torch.save(model.state_dict(), final_weight_path)
    logger.info(f"Final model weights saved to: {final_weight_path}")
    
    writer.flush()
    writer.close()
    
    logger.info("-------Training completed-------")
    if eval_loader is not None:
        logger.info(f"Final validation loss: {eval_all_loss}")
    logger.info(f"Best epoch: {best_epoch}, Best loss: {best_loss}")
