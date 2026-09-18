import torch 
from torch import nn
import math
import torch.nn.functional as F
class CLIPAlignmentBlock(nn.Module):
    def __init__(self,image_dim,text_dim,embed_dim,temperature=0.07):
        super().__init__()
        self.image_dim=image_dim
        self.text_dim=text_dim
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature))
        )

        nn.init.normal_(
            self.image_proj.weight, std=image_dim ** -0.5
        )
        nn.init.normal_(
            self.text_proj.weight, std=text_dim ** -0.5
        )
    def project_image(self, features):
        

        features = self.image_proj(features)
        return F.normalize(
            features.float(), p=2, dim=-1, eps=1e-6
        )

    def project_text(self, features):

        features = self.text_proj(features)

        return F.normalize(
            features.float(), p=2, dim=-1, eps=1e-6
        )

    @staticmethod
    def contrastive_loss(logits_per_image):
        
        ni, nt = logits_per_image.shape

        if ni != nt:
            raise ValueError(
                f"Number of images ({ni}) and texts ({nt}) must be the same."
            )

        labels = torch.arange(
            ni, device=logits_per_image.device
        )
        
        loss_i2t = F.cross_entropy(
            logits_per_image.float(), labels
        )
        loss_t2i = F.cross_entropy(
            logits_per_image.T.float(), labels
        )

        return (loss_i2t + loss_t2i) / 2
    def clamp_logit_scale_(self):
        """
        在 optimizer.step() 后调用。
        将缩放系数限制到不超过 100，避免训练时无限增大。
        """
        self.logit_scale.clamp_(max=math.log(100.0))

    def forward(
        self,
        image_features,
        text_features,
        return_loss=False,
    ):
        image_embeds = self.project_image(image_features)
        text_embeds = self.project_text(text_features)

        logits_per_image = (
            self.logit_scale.exp()
            * (image_embeds @ text_embeds.T)
        )

        outputs = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_image.T,
        }

        if return_loss:
            outputs["loss"] = self.contrastive_loss(
                logits_per_image
            )

        return outputs