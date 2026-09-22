import torch
from torch import nn
import torch.nn.functional as F
import math

class DETR(nn.Module):
    def __init__(
        self,
        num_classes,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pretrained_backbone=False,
        backbone_stride=32,
        aux_loss=True,
    ):
        super().__init__()

        self.backbone = ResNetBackbone(
            pretrained_backbone,
            backbone_stride,
        )

        self.input_projection = nn.Conv2d(
            self.backbone.out_channels,
            d_model,
            kernel_size=1,
        )
        self.position_encoding = PositionEncoding2D(d_model)

        args = (d_model, nhead, dim_feedforward, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(*args)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(*args)
            for _ in range(num_decoder_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        
        self.query_embed = nn.Embedding(num_queries, d_model)

       
        self.class_head = nn.Linear(
            d_model,
            num_classes + 1,
        )

       
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

        self.aux_loss = aux_loss

        
        for stack in (self.encoder, self.decoder):
            for parameter in stack.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def predict(self, hidden):
        return {
            "pred_logits": self.class_head(hidden),
            "pred_boxes": self.box_head(hidden).sigmoid(),
        }

    def forward(self, images, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(
                images.shape[0],
                *images.shape[-2:],
                dtype=torch.bool,
                device=images.device,
            )
        feature = self.input_projection(
            self.backbone(images)
        ) 
        mask = F.interpolate(
            padding_mask[:, None].float(),
            size=feature.shape[-2:],
            mode="nearest",
        )[:, 0].bool()
        pos = self.position_encoding(mask).to(feature.dtype)
        mask = mask.flatten(1)

        memory = feature.flatten(2).transpose(1, 2)

        for layer in self.encoder:
            memory = layer(memory, pos, mask)

        query_pos = self.query_embed.weight.unsqueeze(0).expand(
            images.shape[0],
            -1,
            -1,
        )  # [B,N,D]

        content = torch.zeros_like(query_pos)

        predictions = []

        for layer in self.decoder:
            content = layer(
                content,
                query_pos,
                memory,
                pos,
                mask,
            )

            hidden = self.output_norm(content)
            predictions.append(self.predict(hidden))

        output = predictions[-1]

        if self.aux_loss:
            output["aux_outputs"] = predictions[:-1]

        return output