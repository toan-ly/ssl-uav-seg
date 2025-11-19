import matplotlib.pyplot as plt
import os
import torch
from .utils import compute_iou_sample

def plot_loss(loss_df, save_dir):
    """
    Plot training and validation loss, f1, and IoU curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(loss_df['epoch'], loss_df['train_loss'], label='train_loss')
    axs[0].plot(loss_df['epoch'], loss_df['val_loss'], label='val_loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend(loc='upper right')

    axs[1].plot(loss_df['epoch'], loss_df['train_f1'], label='train_f1')
    axs[1].plot(loss_df['epoch'], loss_df['val_f1'], label='val_f1')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_title('Training and Validation F1 Score')
    axs[1].legend(loc='lower right')

    axs[2].plot(loss_df['epoch'], loss_df['train_iou'], label='train_iou')
    axs[2].plot(loss_df['epoch'], loss_df['val_iou'], label='val_iou')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('IoU')
    axs[2].set_title('Training and Validation IoU')
    axs[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_curve.png'))

@torch.no_grad()
def visualize(model, loader, epoch, save_path=None, num_samples=3, device='cpu'):
    """
    Visualize model predictions on a few samples

    Args:
        model: the trained model
        loader: DataLoader 
        epoch: current epoch number (for title)
        save_path: directory to save the figure
        num_samples: number of samples to visualize
        device: device to run the model on
    """
    model.eval()
    imgs, masks, preds = [], [], []
    for batch in loader:
        img = batch['image'].to(device)
        mask = batch['label'].to(device)

        logits = model(img)
        pred = torch.argmax(logits, dim=1, keepdim=True)

        img = img.as_tensor() if hasattr(img, 'as_tensor') else img
        mask = mask.as_tensor() if hasattr(mask, 'as_tensor') else mask
        pred = pred.as_tensor() if hasattr(pred, 'as_tensor') else pred

        imgs.append(img.cpu())
        masks.append(mask.cpu())
        preds.append(pred.cpu())
        if sum(x.size(0) for x in imgs) >= num_samples:
            break

    imgs = torch.cat(imgs, dim=0)[:num_samples]
    masks = torch.cat(masks, dim=0)[:num_samples]
    preds = torch.cat(preds, dim=0)[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))
    for i in range(num_samples):
        img = imgs[i].permute(1, 2, 0).numpy()

        im_min, im_max = img.min(), img.max()
        if im_max > im_min:
            img = (img - im_min) / (im_max - im_min)

        pred = preds[i].squeeze().numpy()
        mask = masks[i].squeeze().numpy()

        iou = compute_iou_sample(pred, mask)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('GT')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred)
        axes[i, 2].set_title(f'Pred (IoU: {iou:.2f})')
        axes[i, 2].axis('off')

    plt.suptitle(f'Epoch {epoch+1}')
    plt.tight_layout()
    # plt.show()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'epoch_{epoch+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)