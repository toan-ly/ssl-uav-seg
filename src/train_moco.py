import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .moco.builder import MoCoV2
from .datasets import UAVMocoDataset, get_uav_img_paths

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT / 'data' / 'UAVid'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    img_paths = get_uav_img_paths(DATA_DIR)
    dataset = UAVMocoDataset(img_paths, patch_size=224, moco_version=2)

    batch_size = 256 # 256 in original paper
    num_workers = 4

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = MoCoV2(
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        imagenet_init=True,
    ).to(device)

    epochs = 500
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    save_path = ROOT / 'weights' 
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_path / 'moco_v2_resnet50.pth'

    print(f'Number of MoCo parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f'[Resume] Loading checkpoint from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.encoder_q.load_state_dict(checkpoint['encoder_q'])
        model.proj_q.load_state_dict(checkpoint['proj_q'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f'[Resume] Starting from epoch {start_epoch}')

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        start = time.time()
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for im_q, im_k in pbar:
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * im_q.size(0)
            sample_count += im_q.size(0)

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss /= sample_count
        scheduler.step()

        elapsed = time.time() - start
        curr_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| Loss: {epoch_loss:.4f} "
            f"| LR: {curr_lr:.5f} "
            f"| Time: {elapsed:.2f}s")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'encoder_q': model.encoder_q.state_dict(),
                    'proj_q': model.proj_q.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, ckpt_path
            )
            print(f'[Saved] {ckpt_path}')
    
    print('Done training MoCo v2!')

if __name__ == '__main__':
    main()
