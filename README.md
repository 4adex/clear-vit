Vision Transformer (ViT) – Analysis Experiments

This repository presents a series of experiments conducted on a baseline Vision Transformer (ViT) to analyze the effects of different attention mechanisms, positional encoding strategies, and information-theoretic enhancements.

We have completed the image classification task on ImageNet-100 using these variants.
Next, we will extend this work to object detection and semantic segmentation, which are currently in progress.

Overview

We evaluate multiple architectural and attention-level modifications on a fixed ViT backbone.
These include:

1.Shannon entropy–based enhancements

2.Positional encoding modifications

3.Spatial suppression inspired by neuroscience

4.RoPE extensions

5.Hybrid RoPE + Absolute Positional Encoding

All experiments are trained under identical settings to allow fair comparison.
