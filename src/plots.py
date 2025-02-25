import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch.nn.functional as F


def visualize_dataset_samples(dataset, num_samples=3, figsize=(10, 6)):
    fig, ax = plt.subplots(ncols=3, nrows=num_samples, figsize=figsize)
    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        img1, img2, label = dataset[idx]

        img1 = img1.permute(1, 2, 0).numpy()
        img2 = img2.permute(1, 2, 0).numpy()

        ax[i][0].imshow(img1)
        ax[i][1].imshow(img2)
        ax[i][2].text(0.5, 0.5, f"{label}", ha="center", va="center", fontsize=16)

        ax[i][0].axis("off")
        ax[i][1].axis("off")
        ax[i][2].axis("off")

    fig.tight_layout()
    fig.show()


def show_metrics(metrics: dict):
    metric_names = metrics.keys()
    fig, ax_list = plt.subplots(nrows=len(metric_names), figsize=(20, 10))
    for metric_name, ax in zip(metric_names, ax_list):
        ax.plot(metrics[metric_name]["train"], label=f"train {metric_name}", marker="o")
        ax.plot(metrics[metric_name]["val"], label=f"val {metric_name}", marker="s")
        ax.set_title(metric_name)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.show()


def visualize_pair_attention(
    img1,
    img2,
    att_weights,
    mode: Literal["channel", "combined", "spatial"] = "combined",
):
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))

    for i, (img, title) in enumerate(zip([img1, img2], ["Image 1", "Image 2"])):
        ax[0, i].imshow(img.permute(1, 2, 0).clip(0, 1))
        ax[0, i].set_title(title, fontsize=12)
        ax[0, i].axis("off")

    if att_weights.dim() == 4:  # [B,C,H,W]
        att_map = att_weights[0].cpu()
    else:
        att_map = att_weights.cpu()

    # Нормализация
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

    # Режимы визуализации
    if mode == "channel":
        # Канальное внимание (усреднение по пространственным измерениям)
        channel_weights = att_map.mean(dim=(1, 2))
        ax[0, 2].barh(np.arange(len(channel_weights)), channel_weights.numpy())
        ax[0, 2].set_title("Channel Attention Weights", fontsize=12)
        ax[0, 2].set_yticks([])
    else:
        for i, img in enumerate([img1, img2]):
            heatmap = att_map.mean(0) if mode == "combined" else att_map.squeeze()
            heatmap = (
                F.interpolate(
                    heatmap.unsqueeze(0).unsqueeze(0),
                    size=img.shape[1:],
                    mode="bilinear",
                )
                .squeeze()
                .numpy()
            )

            ax[1, i].imshow(img.permute(1, 2, 0).clip(0, 1))
            im = ax[1, i].imshow(heatmap, cmap="jet", alpha=0.6)
            ax[1, i].set_title(f"Attention Map ({mode.capitalize()})", fontsize=12)
            ax[1, i].axis("off")
            fig.colorbar(im, ax=ax[1, i])

    fig.tight_layout()
    fig.show()


def embedding_vis(emb1, shape1, emb2, shape2):
    features = np.concatenate([emb1, emb2], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    tsne_results = tsne.fit_transform(scaled_features)

    plt.figure(figsize=(10, 8))

    labels = np.concatenate([shape1, shape2], axis=0)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            label=f"{'circle' if label else 'square'}",
            alpha=0.6,
            s=25,
            marker="o" if label else "s",
        )

    plt.title("t-SNE визуализация признаков")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.show()
