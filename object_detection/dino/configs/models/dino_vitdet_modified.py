# dino_simple_vit.py

from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .dino_r50 import model  # reuse common structure

# Custom SimpleViT backbone
from detectron2.modeling import SimpleViT

# ViT Params
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

# Replace backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(SimpleViT)(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        dropout=dp,
        attention_dropout=dp,
        use_abs_pos=True,
        use_cls_token=False,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

# Adjust neck input shapes
model.neck.input_shapes = {
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}
model.neck.in_features = ["p3", "p4", "p5", "p6"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4
