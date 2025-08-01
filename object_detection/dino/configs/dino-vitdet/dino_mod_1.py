from detrex.config import get_config
from ..models.dino_vitdet_modified import model
from detectron2.data.datasets import register_coco_instances

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train


register_coco_instances(
    "a-train",
    {},
    "/kaggle/input/coco2017/annotations/instances_train2017.json",
    "/kaggle/input/coco2017/train2017"
)

register_coco_instances(
    "a-val",
    {},
    "/kaggle/input/coco2017/annotations/instances_val2017.json",
    "/kaggle/input/coco2017/val2017"
)

# modify training config
train.init_checkpoint = ""
train.output_dir = "./output/aabb"

# max training iterations
train.max_iter = 90000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

dataloader.train.total_batch_size = 1

dataloader.train.dataset.names = ["a-train"]
dataloader.test.dataset.names = ["a-val"]

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir