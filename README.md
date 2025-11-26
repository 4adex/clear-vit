# **Vision Transformers (ViT)**

This repository presents a series of experiments conducted on a baseline Vision Transformer (ViT) to analyze the effects of different attention mechanisms, positional encoding strategies, and information-theoretic enhancements.

We have completed the image classification task on ImageNet-100 using these variants.
Next, we will extend this work to object detection and semantic segmentation, which are currently in progress.

## **Overview:**

We evaluate multiple architectural and attention-level modifications on a fixed ViT backbone.
These include:

1.Shannon entropy–based enhancements

2.Positional encoding modifications

3.Spatial suppression inspired by neuroscience

4.RoPE extensions

5.Hybrid RoPE + Absolute Positional Encoding

All experiments are trained under identical settings to allow fair comparison.

## **Results:**
| No. | Experiment                                           | acc@1      | acc@5      |
| --- | ---------------------------------------------------- | ---------- | ---------- |
| 1   | Positional Encoding + Dropouts                       | 44.970     | 75.240     |
| 2   | Attention Based on Difference-Similarity             | 46.360     | 75.540     |
| 3   | Baseline ViT                                         | 48.109     | 75.617     |
| 4   | Spatial Suppression Attention                        | 49.383     | 77.673     |
| 5   | RoPE Extension                                       | 53.331     | 80.407     |
| 6   | Differential Attention + Shannon Entropy             | 53.803     | 80.530     |
| 7   | **RoPE Mixed + Absolute Positional Encoding (Best)** | **56.805** | **83.326** |

## **Base Transformer Configuration:**

The following baseline is kept constant for all experiments (unless explicitly noted):

1.Dataset: ImageNet-100

2.Epochs: 90

3.Batch Size: 256

4.Transformer Layers: 12

5.Attention Heads: 3

6.Hidden Dimension: 192

7.MLP Dimension: 768

8.Patch Size: 16×16

9.Baseline Positional Encoding: Learnable


## **Experiment Details:**


### **1. Baseline ViT:**

A standard ViT with learnable positional embeddings and vanilla multi-head attention.
Serves as the primary reference for all comparisons.


### **2. Positional Encodings with Dropouts:**

Applying dropout on positional embeddings to test robustness against spatial perturbations.


### **3. Attention Based on Difference Similarity:**

Uses similarity of feature differences (ΔQ, ΔK) rather than raw features.

Key Idea

  ΔQi=Qi−Qi−1 ,  ΔKj​=Kj−Kj−1

Attention scores:

  <img width="245" height="97" alt="image" src="https://github.com/user-attachments/assets/dbee582a-ace3-4fd2-aa1c-57592963eb3f" />

  Enhances local relational modeling but loses some global context.

  
  ### **4. Differential Attention with Shannon Entropy:**

Replaces the learnable scalar λ in differential attention with patch-dependent Shannon entropy:
   H(x)=−∑p(x)logp(x)

   High-entropy patches receive stronger modulation → better handling of texture-rich or complex regions.

Significant gain over the baseline.


### **5. Spatial Suppression Attention:**

Inspired by surround suppression in biological vision.

A depthwise convolution learns a suppression kernel per head and subtracts it from raw attention scores:
	suppressed = scores - self.suppression_conv(scores)
​Removes noisy neighborhood interactions → improves attention quality.


### **6. RoPE Extension:**

Unlike standard RoPE that rotates Q and K independently:

This version rotates Q–K pairs relative to each other


### **7. RoPE Mixed + Absolute Positional Encoding (Best Model):**

The best-performing configuration.

Combines:

  -Absolute Positional Encoding (APE) for global positioning

  -Modified RoPE for fine-grained relative geometric cues

Balances global + local spatial understanding → highest overall accuracy.

Rotation angle depends on query–key positional relationship

Enables richer relative geometry modeling

Shows strong improvement over baseline.



### ✅ **Completed :**

Classification module on ImageNet-100

Multiple positional and attention mechanism experiments

### 🔄 **In Progress :**

Object Detection (ViT backbone)                         

Semantic Segmentation (ViT encoder with segmentation head)

Results and implementations will be added soon.


