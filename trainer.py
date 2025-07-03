from dataloader import train_dataset, LabelSmoothing, train_loader, val_loader
from models import SimpleVisionTransformer, SSSVisionTransformer
import torch.nn as nn
import torch.distributed as dist
import torch
import os
import shutil
from enum import Enum
import time
import logging
import datetime

# Setup logging
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = os.path.join(log_dir, f"training_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will also print to console
    ]
)

num_epochs = 90
batch_size = 256

weight_decay = 0.05
learning_rate = 1e-4

log_steps = 2500
start_step = 0

checkpoint_path = "./checkpoints"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n = len(train_dataset)
total_steps = round((n * num_epochs) / batch_size)
warmup_try = 5000

def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": total_params - trainable_params
    }

def save_checkpoint(state, is_best, path, filename='imagenet_baseline_patchconvcheckpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

def save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path):
    # Define the filename with the current step
    filename = os.path.join(checkpoint_path, f'BaseLine_VIT.pt')
    
    # Save the checkpoint
    torch.save({
        'step': step,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, filename)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



# model = SimpleVisionTransformer(
#     image_size=256,
#     patch_size=16,
#     num_layers=12,
#     num_heads=3,
#     hidden_dim=192,
#     mlp_dim=768,
# )
model = SSSVisionTransformer(
    image_size=256,
    patch_size=16,
    num_layers=12,
    num_heads=3,
    hidden_dim=192,
    mlp_dim=768,
)
model = nn.DataParallel(model)
model.to('cuda')

wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]
original_model = model

# =========== Label smoothing loss ==============
criterion = LabelSmoothing(smoothing=0.1)
criterion.to(device)


# ============ Optimizer =====================
optimizer = torch.optim.AdamW(
    [
        {"params": wd_params, "weight_decay": weight_decay},
        {"params": non_wd_params, "weight_decay": weight_decay},
    ],
    lr=learning_rate,
    betas=(0.9, 0.999)  # Set beta1=0.9 and beta2=0.999
)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / warmup_try)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_try)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_try])


logging.info(f"Model Parameters: {count_parameters(model)}")


# ==============================================
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, step, use_wandb=False, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            elif torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    progress.display_summary()

    return top1.avg

def train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    print_freq = 100
    log_steps = 2500

    best_acc1 = 0.0
    
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, losses, top1, top5]
    )

    model.train()
    end = time.time()
    
    def infinite_loader():
        while True:
            yield from train_loader
            
    for step, (images, labels_a, labels_b, lam) in zip(range(start_step + 1, total_steps + 1), infinite_loader()):
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        labels_a = labels_a.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)
        
        # Convert lam to a tensor if it's not already one
        if not isinstance(lam, torch.Tensor):
            lam = torch.tensor(lam, device=device)
        else:
            lam = lam.to(device, non_blocking=True)

        output = model(images)
        loss = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)

        # Compute accuracy (this is an approximation for mixed labels)
        acc1_a, acc5_a = accuracy(output, labels_a, topk=(1, 5))
        acc1_b, acc5_b = accuracy(output, labels_b, topk=(1, 5))
        acc1 = lam * acc1_a + (1 - lam) * acc1_b
        acc5 = lam * acc5_a + (1 - lam) * acc5_b

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        loss.backward()
        l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if step % print_freq == 0:
            progress.display(step)
        
        if ((step % print_freq == 0) and ((step % log_steps != 0) and (step != total_steps))):        
            save_checkpoint_step(step, model, best_acc1, optimizer, scheduler, checkpoint_path)
                
        if step % log_steps == 0:
            acc1 = validate(val_loader, original_model, criterion, step)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_path)
            
        elif step == total_steps:
            acc1 = validate(val_loader, original_model, criterion, step)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            
            save_checkpoint({
                'step': step,
                'state_dict': original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint_path)
            

        scheduler.step()


train(train_loader, val_loader, start_step, total_steps, original_model, model, criterion, optimizer, scheduler, device)
