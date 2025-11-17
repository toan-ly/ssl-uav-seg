import time
from pure_eval import Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from monai.losses import (
    DiceLoss, FocalLoss, TverskyLoss, 
    HausdorffDTLoss, DiceCELoss
)
from .utils import *
from .metric import *
from monai.inferers import sliding_window_inference

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        loss='dice',
        optimizer_name='adamw',
        lr=1e-4,
        early_stopping=True,
        patience=3,
        scheduler=None,
        cls_weights=None,
        num_classes=8,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_name = loss
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.scheduler_name = scheduler

        self.cls_weights = None
        if cls_weights:
            self.cls_weights = torch.tensor(cls_weights, dtype=torch.float32).to(self.device)

        self.criterion = self._get_loss(loss)
        self.optimizer = self._get_optimizer(optimizer_name, lr)
        self.scheduler = None
        if scheduler:
            self.scheduler = self._get_scheduler(scheduler)

      
        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_iou = -float('inf')
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_weights = None

        self.checkpoint = {
            'train_loss': [],
            'train_iou': [],
            'train_f1': [],
            'val_loss': [],
            'val_iou': [],
            'val_f1': [],
        }

        self.train_evaluator = Evaluator(num_class=num_classes)
        self.val_evaluator = Evaluator(num_class=num_classes)

    def train_one_epoch(self, epoch, epochs):
        self.model.train()
        epoch_loss = 0.0
        desc = f"Epoch [{epoch+1}/{epochs}] Training"

        for batch in tqdm(self.train_loader, desc=desc, leave=False):
            imgs = batch['image'].to(self.device) # [B, C, H, W]
            masks = batch['label'].to(self.device) # [B, 1, H, W]

            self.optimizer.zero_grad()

            logits = self.model(imgs) # [B, num_cls, H, W] 
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1, keepdim=True) 

            for i in range(masks.shape[0]):
                self.train_evaluator.add_batch(
                    masks[i].cpu().numpy(), 
                    preds[i].cpu().numpy()
                )

        miou = np.nanmean(self.train_evaluator.Intersection_over_Union())
        oa = np.nanmean(self.train_evaluator.OA())
        f1 = np.nanmean(self.train_evaluator.F1())
        eval_dict = {
            'mIoU': miou,
            'OA': oa,
            'f1': f1
        }


        epoch_loss = epoch_loss / len(self.train_loader.dataset)

        self.train_evaluator.reset()

        return epoch_loss, eval_dict
    
    @torch.no_grad()
    def validate(self, epoch, epochs):
        self.model.eval()
        epoch_loss = 0.0
        desc = f"Epoch [{epoch+1}/{epochs}] Validation"
        
        for batch in tqdm(self.val_loader, desc=desc, leave=False):
            imgs = batch['image'].to(self.device) # [B, C, H, W]
            masks = batch['label'].to(self.device) # [B, 1, H, W]

            logits = sliding_window_inference(
                imgs,
                roi_size=(512, 512),
                sw_batch_size=2,
                predictor=self.model,
                overlap=0.25,
            )
            
            loss = self.criterion(logits, masks)

            epoch_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1, keepdim=True) 
            for i in range(masks.shape[0]):
                self.val_evaluator.add_batch(
                    masks[i].cpu().numpy(), 
                    preds[i].cpu().numpy()
                )

        miou = np.nanmean(self.val_evaluator.Intersection_over_Union())
        oa = np.nanmean(self.val_evaluator.OA())
        f1 = np.nanmean(self.val_evaluator.F1())
        eval_dict = {
            'mIoU': miou,
            'OA': oa,
            'f1': f1
        }

        epoch_loss = epoch_loss / len(self.val_loader.dataset)

        self.val_evaluator.reset()
        return epoch_loss, eval_dict

    def fit(self, epochs=10, verbose=True, save_model_path=None, save_plots_path=None):
        start_time = time.time()
        for epoch in range(epochs):
            train_loss, train_eval = self.train_one_epoch(epoch, epochs)
            val_loss, val_eval = self.validate(epoch, epochs)

            self.checkpoint['train_loss'].append(train_loss)
            self.checkpoint['train_iou'].append(train_eval['mIoU'])
            self.checkpoint['train_f1'].append(train_eval['f1'])
            self.checkpoint['val_loss'].append(val_loss)
            self.checkpoint['val_iou'].append(val_eval['mIoU'])
            self.checkpoint['val_f1'].append(val_eval['f1'])

            if verbose:
                print(
                    f'Epoch {epoch+1}/{epochs} '
                    f'| Train Loss: {train_loss:.4f}, F1: {train_eval["f1"]:.4f}, IoU: {train_eval["mIoU"]:.4f} '
                    f'| Val Loss: {val_loss:.4f}, F1: {val_eval["f1"]:.4f}, IoU: {val_eval["mIoU"]:.4f}'
                )
            if verbose and save_plots_path and (epoch + 1) % 5 == 0:
                visualize(
                    self.model, 
                    self.val_loader, 
                    epoch, 
                    save_path=save_plots_path, 
                    num_samples=3, 
                    device=self.device)

            if val_eval["mIoU"] > self.best_iou:
                self.best_iou = val_eval["mIoU"]

            # Early stopping
            if self.early_stopping:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.epochs_no_improve = 0
                    self.best_weights = self.model.state_dict()
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print("=> Early stopping at epoch", epoch+1)
                        if self.best_weights is not None:
                            self.model.load_state_dict(self.best_weights)
                        break

            if self.scheduler:
                self.scheduler.step()

        # Load best weights
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        total_time = time.time() - start_time
        print(f'Training time: {total_time:.2f}s | (best val miou = {self.best_iou:.4f})')

        # Save model
        if save_model_path:
            self._save_checkpoint(save_model_path)
            print(f'Model saved to {save_model_path}')

        return self.checkpoint
      
    def _get_optimizer(self, optim_name='adamw', lr=1e-4, weight_decay=1e-2):
        if optim_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if optim_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if optim_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f'Unknown optimizer: {optim_name}')    

    def _get_scheduler(self, scheduler_name='cosine'):
        scheduler_map = {
            'step': optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5),
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6),
        }
        if scheduler_name not in scheduler_map:
            raise ValueError(f'Unknown scheduler: {scheduler_name}')
        return scheduler_map[scheduler_name]

    def _get_loss(self, name):
        """
        Examples of loss_name:
            'dicece' (default)
            'bce' 
            'dice'
            'bce+dice'
            'bce+dice@0.3,0.7'
            'tversky+focal+hausdorff@0.45,0.45,0.1'
        """
        criterions = {
            'bce': lambda: nn.BCEWithLogitsLoss(weight=self.cls_weights),
            'dice': lambda: DiceLoss(softmax=True, squared_pred=True, to_onehot_y=True),
            'focal': lambda: FocalLoss(
                gamma=2.0,
                use_softmax=True,
                to_onehot_y=True,
                weight=self.cls_weights
            ),
            'tversky': lambda: TverskyLoss(
                softmax=True, 
                to_onehot_y=True,
                alpha=0.3, beta=0.7
            ),
            'hausdorff': lambda: HausdorffDTLoss(
                softmax=True, 
                include_background=True, 
                to_onehot_y=True
            ),
            'dicece': lambda: DiceCELoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                weight=self.cls_weights
            )
        }

        name = name.strip().lower()

        if '+' in name:
            if '@' in name:
                loss_part, weight_part = name.split('@')
                weights = [float(w) for w in weight_part.split(',')]
            else:
                loss_part = name
                weights = None
            loss_names = loss_part.split('+')
            if loss_names[0] not in criterions or loss_names[1] not in criterions:
                raise ValueError(f'Unknown loss(es): {loss_names[0]}, {loss_names[1]}')
            
            w1, w2 = (0.5, 0.5) if weights is None else (weights[0], weights[1])
            criterion1 = criterions[loss_names[0]]()
            criterion2 = criterions[loss_names[1]]()

            return lambda preds, targets: w1 * criterion1(preds, targets) + w2 * criterion2(preds, targets)
        
        if name not in criterions:
            raise ValueError(f'Unknown loss: {name}')
        return criterions[name]()
        
    def _save_checkpoint(self, path):
        """
        Save model checkpoint including model state, config, and training configs
        """
        model_config = getattr(self.model, 'model_config', None)
        ckpt = {
            'state_dict': self.model.state_dict(),
            'model_config': model_config,
            'n_params': self.n_params,
            'train_config': {
                'loss': self.loss_name,
                'optimizer': self.optimizer_name,
                'lr': self.lr,
                'scheduler': self.scheduler_name,
            }
        }
        torch.save(ckpt, path)