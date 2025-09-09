# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:  
# Author:       Sedigheh Eslami 
# Date:         2021/08/06
#-------------------------------------------------------------------------------
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


def _convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

class CLIPWithDropout(nn.Module):
    def __init__(self, clip_model, p=0.5):
        super().__init__()
        self.clip_model = clip_model
        self.dropout = nn.Dropout(p)

    def forward(self, images, captions):
        
        # Encode separately
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(captions)

        # Apply dropout
        image_features = self.dropout(image_features)
        text_features = self.dropout(text_features)

        # Normalize as CLIP usually does
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # CLIP similarity calculation
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# Train phase
def train(cfg, train_loader, eval_loader, device):
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir)
    logger, _ = create_logger(cfg)
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logger.info(f"-------Loading CLIP with vision encoder {cfg.TRAIN.VISION_ENCODER} -------")
    base_model, preprocess = clip.load(cfg.TRAIN.VISION_ENCODER, device=device, jit=False)

    dropout_prob = getattr(cfg.TRAIN, 'DROPOUT_PROB', 0.5)
    model = CLIPWithDropout(base_model, p=dropout_prob).to(device)

    if device == "cpu":
          model.float()
    else :
        clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optim = get_optimizer(cfg, model)

    best_loss, best_epoch, best_model = 10000, 0, ""

    early_stopping_patience = 3
    early_stopping_counter = 0

    logger.info("-------Training started-------")
    logger.info(f"-------Early stopping enabled with patience: {early_stopping_patience}-------")

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
        current_loss = None
        if eval_loader is not None:
            eval_all_loss = valid_model(eval_loader, model, loss_img, loss_txt, cfg, device)
            current_loss = eval_all_loss

            if eval_all_loss < best_loss:
                best_loss = eval_all_loss
                best_epoch = epoch
                best_model = model
                early_stopping_counter = 0
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
            else:
                early_stopping_counter += 1

            logger.info(
                    f"--------------Epoch:{epoch}    Eval_Loss:{eval_all_loss}%--------------")
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Eval_Loss:{best_loss}%--------------")
            logger.info(f"--------------Early Stopping Counter: {early_stopping_counter}/{early_stopping_patience}--------------")
            writer.add_scalar("Loss/val", eval_all_loss, epoch)
        else:
            current_loss = train_all_loss

            if train_all_loss < best_loss:
                best_loss = train_all_loss
                best_epoch = epoch
                best_model = mode
                early_stopping_counter = 0
                torch.save({
                    'best_epoch': best_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_loss': best_loss,
                    }, os.path.join(model_dir, "best_model.pth")
                    )
            else:
                early_stopping_counter += 1
            logger.info(
                    f"--------------Best_Epoch:{best_epoch}    Best_Train_Loss:{best_loss}%--------------")
            logger.info(f"--------------Early Stopping Counter: {early_stopping_counter}/{early_stopping_patience}--------------")
            writer.add_scalar("Loss/train-as-val", train_all_loss, epoch)

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"-------Early stopping triggered! No improvement for {early_stopping_patience} epochs-------")
            logger.info(f"-------Training stopped at epoch {epoch}-------")
            logger.info(f"-------Best model was saved at epoch {best_epoch} with loss {best_loss}-------")
            break

        if not os.path.exists(cfg.RESULTS_DIR):
            os.makedirs(cfg.RESULTS_DIR)

        with open(os.path.join(cfg.RESULTS_DIR, "best.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_loss": best_loss, "stopped_early": early_stopping_counter >= early_stopping_patience, "final_epoch": epoch}, f)
        writer.flush()
        writer.close()

