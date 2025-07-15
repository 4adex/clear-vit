# train_simple_vit.py

from detrex.config import get_config
from ..models.dino_vitdet_modified import model

# Common defaults
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# Custom training parameters
train.init_checkpoint = ""  # No pretrained weights
train.output_dir = "./output/dino_simple_vit"

train.max_iter = 90000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# Optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# Dataloader config
dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
