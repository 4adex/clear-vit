import torch

# Load the classification-trained checkpoint
checkpoint = torch.load("./checkpoints/BaseLine_VIT.pt", map_location="cpu")

# Extract only the state_dict from 'model.module'
state_dict = checkpoint["state_dict"]

# Remove 'module.' prefix from keys
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Save the cleaned state_dict for Detectron2
torch.save(clean_state_dict, "./checkpoints/simple_vit_backbone_pretrain.pth")
