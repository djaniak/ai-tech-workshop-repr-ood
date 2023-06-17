from typing import Union

import torch
import torch.nn.functional as F
import tqdm
from pytorch_lightning import LightningModule
from torch.utils.data.dataloader import DataLoader


def get_preds_and_labels(model, dataloader):
    target = []
    preds = []
    for batch in tqdm.tqdm(dataloader):
        x, y = batch
        with torch.no_grad():
            logits = model(x)
        preds.append(F.softmax(logits, dim=1))
        target.append(y)
    preds = torch.vstack(preds)
    target = torch.hstack(target)
    return preds, target


def extract_features_from_simclr(
    model: LightningModule, dataloader: DataLoader, device: Union[torch.device, str]
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    model = model.to(device)
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        x, y = batch
        with torch.no_grad():
            batch_feats = model(x.to(device))
        features.append(batch_feats.cpu())
        labels.append(y)
    features = torch.vstack(features)
    labels = torch.hstack(labels)
    return features, labels
