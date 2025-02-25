from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for img1, img2, labels in dataloader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(img1, img2).squeeze()
        smoothed_labels = labels * 0.9 + 0.05
        loss = criterion(logits, smoothed_labels)

        loss.backward()
        total_norm = (
            nn.utils.clip_grad_norm_(model.transformer.parameters(), 1.0)
            if hasattr(model, "transformer")
            else nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
        total_loss += loss.item()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc, f1, total_norm


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            logits = model(img1, img2).squeeze()
            loss = criterion(logits, labels.to(torch.float32))

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc, f1
