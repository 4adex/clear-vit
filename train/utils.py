def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": total_params - trainable_params
    }
