import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MoCoV2(nn.Module):
    def __init__(
        self,
        base_encoder=models.resnet50,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        imagenet_init: bool = True,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        imagenet_init: whether to initialize with ImageNet pretrained weights (default: True)
        """
        super(MoCoV2, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        if imagenet_init:
            backbone_q = base_encoder(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_k = base_encoder(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone_q = base_encoder(weights=None)
            backbone_k = base_encoder(weights=None)

        # Encoder query
        dim_mlp = backbone_q.fc.in_features
        backbone_q.fc = nn.Identity()
        self.encoder_q = backbone_q

        self.proj_q = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim),
        )

        # Encoder key
        backbone_k.fc = nn.Identity()
        self.encoder_k = backbone_k

        self.proj_k = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim),
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue (memory bank)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        encoder_k = m * encoder_k + (1 - m) * encoder_q
        proj_k = m * proj_k + (1 - m) * proj_q
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.proj_q.parameters(), self.proj_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # Compute query features
        q = self.encoder_q(im_q)  
        q = self.proj_q(q)  
        q = F.normalize(q, dim=1)  

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  

            k = self.encoder_k(im_k)  
            k = self.proj_k(k)  
            k = F.normalize(k, dim=1)  

        # Compute logits
        # positive logits: q dot k, Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: q dot queue, NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return tensor
