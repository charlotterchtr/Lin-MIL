# Lin-MIL

[OpenReview](https://openreview.net/forum?id=Cbzq1wDu2v)

Code repository for paper **"Linear Attention for Whole Slide Image Analysis"** published at **MICCAI 2025** *COMPAYL workshop*.

-----------------------------

### Abstract

Deep learning–based analysis of gigapixel whole slide images (WSIs) in computational pathology (CPath) typically relies on patch-level feature extraction and instance aggregation, with attention-based contextualization at the core of state-of-the-art methods. However, scalability is a major challenge due to the vast number of patches. Therefore, we introduce linear attention based multiple-instance learning (Lin-MIL), which transposes and interchanges the calculations of queries, keys, and values in the attention mechanism. By leveraging linear attention, Lin-MIL reduces computational complexity from O(n²d) to O(nd²), compared to vanilla self-attention. Despite this efficiency gain, Lin-MIL outperforms 12 baseline methods across biomarker, mutation, and tumor classification benchmarks, while also demonstrating robust out-of-domain performance. Moreover, its qualitative attention maps highlight diagnostically relevant regions. In summary, Lin-MIL provides increased performance as well as enhanced scalability and interpretability for a range of computational pathology tasks.

**Lin-MIL Architecture:**

![Lin-MIL Architecture](./figures/lin_mil_architecture.png)
*Figure 1: Lin-MIL pipeline for WSI analysis. (A) In the feature embedding stage, we tesselate the WSI after background removal and extract patch-level features using a pathology FM. (B) The Lin-MIL architecture shrinks the latent dimension by a projection layer, and aggregates the sequence by linear attention blocks followed by pooling and a classification head. (C) Each linear attention block calculates linear attention followed by normalization and a multilayer perceptron.*

**Linear Attention:**

![Lin-MIL Attention Ablation](./figures/lin_mil_attention.png)
*Figure 2: Comparison of attention mechanisms. (A) Vanilla softmax attention calculates attention scores by multiplying queries Q and keys K in O(n²d), followed by softmax weighting and multiplication with values V. (B) Nyström Attention approximates self-attention by incorporating rank reduction to achieve a complexity of O(nm), with landmarks m ≪ n. (C) Dilated Attention reduces the number of operations by varying dilation ratios to O(n). (D) Linear Attention uses a decomposable kernel function φ(·) = ReLU(·), to first calculate φ(K^T)×V and then obtain the attention-weighted values V' in O(nd²).*


### Installation

You can easily download and install the Lin-MIL model via:

```bash
pip install lin_mil
```

Or you can install the package in editable mode via:

```bash
git clone https://github.com/charlotterchtr/Lin-MIL.git
cd Lin-MIL
pip install -e .
```

### Data Preprocessing

For whole slide image segmentation and tesselation into patches, we used the [CLAM library](https://github.com/mahmoodlab/CLAM) and the [UNI](https://github.com/mahmoodlab/UNI) foundation model was used for feature extraction.

### Usage

```python
import torch
from lin_mil import Lin_MIL, Config

# simulate data
batch_size = 1      # number of WSIs in a batch
num_tiles = 1000    # number of patches per WSI
patch_dim = 1024    # feature dimension of each patch, depending on foundation model
num_classes = 2     # number of targets to predict
wsi = torch.randn(batch_size, num_tiles, patch_dim)

# initialize model parameters
config = Config()

# required
config.num_classes = num_classes
config.input_dim = patch_dim

# optional
config.latent_dim = 512
config.transformer_depth = 4
config.dropout = 0.0
config.emb_dropout = 0.1
config.act = 'ReLU'           # options: ReLU, GeLU, LeakyReLU, ELU, TanH, Softplus
config.attention =  'linear'
config.pooling = 'cls'        # options: cls, mean
config.ablation = False

# init
model = Lin_MIL(num_classes=num_classes, input_dim=patch_dim, config=config)

# forward pass
logits, attn = model(wsi) 
# logits shape: (batch_size, num_classes)
# attn shape: (batch_size, num_tiles)

# probabilities for multiclass classification
probs = torch.softmax(logits, dim=1)
```

You can also compare the performance of different attention mechanisms. Please note that dilated attention requires flash-attn and xformers to be installed.

```python

# ablation on attention mechanism
config.ablation = True
config.attention = 'nystrom' # options: 'linear', 'nystrom', 'softmax', 'dilated'

model_ablation = Lin_MIL(num_classes=num_classes, input_dim=patch_dim, config=config)

logits_ablation, attn_ablation = model_ablation(wsi)
probs_ablation = torch.softmax(logits_ablation, dim=1)
```
