# CS5330 Project: Query-Driven Cross-Attention for Multi-View Feature Fusion in BEV Perception

## Overview
A deep learning approach that enhances Cam2BEV with query-driven cross-attention for improved Bird's Eye View (BEV) perception from multiple camera views. The model achieves superior performance in BEV semantic segmentation by dynamically fusing features from different camera perspectives.

## Key Features
- PyTorch implementation of Cam2BEV architecture
- Query-driven cross-attention mechanism for adaptive feature fusion
- Improved spatial feature retention compared to global pooling approaches
- Support for multi-view camera inputs with 360-degree coverage

## Model Architecture

### Baseline: Cam2BEV
The baseline processes multiple camera views through:
- Independent encoder paths for each camera input
- Feature map transformation using IPM
- U-Net style architecture with concatenation-based fusion

### Enhanced Architecture
- **Query-Driven Cross-Attention**: Uses learnable queries to adaptively fuse multi-view features
- **Patch-Level Processing**: Maintains spatial resolution through patch-level feature representation
- **Positional Encoding**: Incorporates sinusoidal positional encodings for spatial context

## Dataset
- Virtual Test Drive (VTD) simulation platform
- 8,000 training samples
- 500 testing samples
- 10 semantic categories
- 4 wide-angle camera views


Here's the Python code to download the specified Kaggle dataset:

```python
import kaggle
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the dataset
kaggle.api.dataset_download_files(
    'suryajrrafl/cam2bev-frlr',
    path='./data',
    unzip=True
)
```

Before running this code, you need to:

1. Install the Kaggle package:
```bash
pip install kaggle
```

2. Set up your Kaggle API credentials:
- Go to your Kaggle account settings
- Click on "Create New API Token" to download `kaggle.json`
- Place the `kaggle.json` file in:
  - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
  - Linux: `~/.kaggle/kaggle.json`
  - macOS: `/Users/<username>/.kaggle/kaggle.json`


## Training Configuration

| Parameter | Value |
|-----------|--------|
| Learning Rate | 3e-3 |
| Optimizer | Adam |
| Batch Size | 32 |
| Training Epochs | 50 |
| Weight Decay | 0.01 |
| LR Schedule | Cosine Annealing |
| Input Resolution | 64×128 |
| BEV Resolution | 64×128 |[1]

## Performance Results

| Model Variant | mIoU |
|--------------|------|
| Baseline Cam2BEV | 0.63 |
| With Global-Average Pooling | 0.59 |
| With Cross-Attention | 0.68 |

## Training Characteristics
- Initial convergence comparable to baseline
- Temporary performance dip (epochs 15-25)
- Gradual recovery and improvement
- Stable performance above baseline

## Future Work
- Integration of temporal dynamics
- Optimization of attention mechanism
- Enhanced real-time performance
- Improved scalability for diverse environments

## License
MIT License

## Citation
```
@article{tangri2024query,
  title={Query-Driven Cross-Attention for Multi-View Feature Fusion in Bird's Eye View Perception},
  author={Tangri, Arsh and Singh, Aditya},
  year={2024}
}
```