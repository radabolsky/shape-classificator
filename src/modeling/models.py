from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid(),
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        gap1 = F.adaptive_avg_pool2d(x1, 1)
        gap2 = F.adaptive_avg_pool2d(x2, 1)
        channel_weights = self.channel_att(
            torch.cat([gap1.view(x1.size(0), -1), gap2.view(x2.size(0), -1)], dim=1)
        ).view(x1.size(0), -1, 1, 1)

        max_pool, _ = torch.max(torch.cat([x1, x2], dim=1), dim=1, keepdim=True)
        avg_pool = torch.mean(torch.cat([x1, x2], dim=1), dim=1, keepdim=True)
        spatial_weights = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))

        combined_att = channel_weights * spatial_weights
        return x1 + self.gamma * (combined_att * x2)

    def compute_attention(self, x1, x2):
        gap1 = F.adaptive_avg_pool2d(x1, 1)
        gap2 = F.adaptive_avg_pool2d(x2, 1)
        channel_weights = self.channel_att(
            torch.cat([gap1.flatten(1), gap2.flatten(1)], dim=1)
        ).view(*x1.shape[:2], 1, 1)  # [B, C, 1, 1]

        max_pool, _ = torch.cat([x1, x2], dim=1).max(dim=1, keepdim=True)
        avg_pool = torch.cat([x1, x2], dim=1).mean(dim=1, keepdim=True)
        spatial_weights = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))

        return channel_weights * spatial_weights

    def get_attention_map(
        self, feat1, feat2, mode: Literal["channel", "combined", "spatial"] = "combined"
    ):
        with torch.no_grad():
            gap1 = F.adaptive_avg_pool2d(feat1, 1)
            gap2 = F.adaptive_avg_pool2d(feat2, 1)
            channel_weights = self.channel_att(
                torch.cat([gap1.flatten(1), gap2.flatten(1)], dim=1)
            )

            max_pool = torch.cat([feat1, feat2], 1).max(dim=1, keepdim=True)[0]
            avg_pool = torch.cat([feat1, feat2], 1).mean(dim=1, keepdim=True)
            spatial_weights = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))

            if mode == "channel":
                return channel_weights.view(-1, 1, 1)  # [B, 1, 1]
            elif mode == "spatial":
                return spatial_weights  # [B, 1, H, W]
            else:
                return channel_weights.view(*feat1.shape[:2], 1, 1) * spatial_weights


class AGFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("resnet50")
        self.feature_extractor = nn.Sequential(*list(model.children())[:5])
        self.projection = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.attn_gate = AttentionGate(256)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.classifier = nn.Linear(1, 1)

    def forward(self, x1, x2):
        emb1, emb2 = self.get_embeddings(x1, x2)

        sim = self.similarity(emb1, emb2)

        return self.classifier(sim.unsqueeze(1)).squeeze(1)

    def get_embeddings(self, img1, img2):
        f1 = self.feature_extractor(img1)
        f2 = self.feature_extractor(img2)

        f1_attn = self.attn_gate(f1, f2)
        f2_attn = self.attn_gate(f2, f1)

        g1 = F.adaptive_avg_pool2d(f1_attn, (1, 1)).squeeze()
        g2 = F.adaptive_avg_pool2d(f2_attn, (1, 1)).squeeze()
        emb1 = self.projection(g1)
        emb2 = self.projection(g2)
        return emb1, emb2


class DualStreamFusion(nn.Module):
    def __init__(self, cnn_model: nn.Module):
        super().__init__()
        self.cnn = cnn_model
        self.cnn.fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.cnn.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(512, 1)

    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        fused = torch.cat([feat1, feat2], dim=1)
        output = self.fusion(fused)
        return self.classifier(output)
