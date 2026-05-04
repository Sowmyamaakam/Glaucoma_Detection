# Glaucoma Detection using HACapsNet

A deep learning project for detecting glaucoma in fundus images using Hybrid Attention Capsule Networks with advanced medical image segmentation and classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Project Architecture](#project-architecture)
- [Visualizations](#visualizations)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Features](#key-features)

---

## 🔍 Overview

This project implements a **Hybrid Attention Capsule Network (HACapsNet)** for:
- **Classification**: Detecting glaucoma vs normal eyes
- **Segmentation**: Localizing optic disc and cup regions
- **Interpretability**: Providing visual explanations via Grad-CAM and Saliency maps

**Key Technologies:**
- PyTorch for model implementation
- Capsule Networks for hierarchical feature learning
- Focal Loss for handling class imbalance
- ResNet18 backbone with attention mechanisms
- CLAHE preprocessing for image enhancement

---

## 📁 Dataset Structure

```
Glaucoma_Detection/
├── README.md
├── glaucoma_code.py
├── images/
│   ├── normal/
│   │   ├── normal_001.jpg
│   │   ├── normal_002.jpg
│   │   └── ... (non-glaucoma fundus images)
│   └── glaucoma/
│       ├── glaucoma_001.jpg
│       ├── glaucoma_002.jpg
│       └── ... (glaucoma fundus images)
├── visualizations1/
│   ├── 01_training_curves.png
│   ├── 02_confusion_matrix.png
│   ├── 03_roc_curve.png
│   ├── 04_precision_recall_curve.png
│   ├── 05_class_distribution.png
│   ├── 06_metrics_comparison.png
│   ├── 07_segmentation_results.png
│   ├── 08_segmentation_overlay.png
│   ├── 09_gradcam_visualization.png
│   ├── 10_saliency_maps.png
│   ├── 11_error_analysis.png
│   ├── 12_capsule_activations.png
│   ├── 13_dice_distribution.png
│   ├── 14_feature_maps_sample_1.png
│   ├── 14_feature_maps_sample_2.png
│   ├── 15_metrics_table.png
│   ├── 16_training_time_analysis.png
│   ├── 17_all_metrics_progression.png
│   ├── test_metrics.csv
│   └── training_history.csv
└── models/
    ├── best_model.pth
    └── final_model.pth
```

---

## 🏗️ Project Architecture

### HACapsNet Model Components

```
Input Image (3, 512, 512)
        ↓
    ResNet18 Encoder
        ↓
    ┌───────────────────┐
    │   Feature Maps    │
    │    (512, 16, 16)  │
    └─────────┬─────────┘
              ↓
    ┌─────────────────────────────────┐
    │  Branch 1: Segmentation Path    │
    │  Decoder Network with           │
    │  ConvTranspose2D Layers         │
    └──────────┬──────────────────────┘
               ↓
        Segmentation Output
         (Optic Disc & Cup)
        
        ┌──────────────────────────────┐
        │  Branch 2: Classification    │
        │  Primary Capsules (8)        │
        │  Routing Layer               │
        │  Digit Capsules (2 classes)  │
        └──────────┬───────────────────┘
                   ↓
          Classification Output
           (Glaucoma or Normal)
```

### Key Layers

| Component | Details |
|-----------|---------|
| **Encoder** | ResNet18 (pretrained on ImageNet) |
| **Primary Capsules** | 8 capsules, 16-dimensional vectors |
| **Routing Algorithm** | Dynamic routing by agreement (3 iterations) |
| **Digit Capsules** | 2 capsules (Normal/Glaucoma) |
| **Decoder** | 3× ConvTranspose2D layers for segmentation |
| **Loss Function** | Focal Loss (α=3.0, γ=2.0) + BCEWithLogits |

---

## 📊 Visualizations (17 Research-Grade Graphs)

### 1. **Training Performance Curves** (01_training_curves.png)
6-panel visualization showing:
- Total Training Loss progression
- Segmentation Loss (green)
- Classification Loss (red)
- Training Accuracy curve
- F1-Score progression
- Dice Score improvement

**Use Case**: Monitor model convergence and identify overfitting

---

### 2. **Confusion Matrix** (02_confusion_matrix.png)
Heatmap showing:
- True Positives (TP): Correctly identified glaucoma cases
- True Negatives (TN): Correctly identified normal cases
- False Positives (FP): Normal misclassified as glaucoma
- False Negatives (FN): Glaucoma misclassified as normal

**Use Case**: Identify classification imbalances and sensitivity

---

### 3. **ROC Curve** (03_roc_curve.png)
- Shows True Positive Rate vs False Positive Rate
- AUC-ROC score for overall model performance
- Comparison with random classifier baseline

**Use Case**: Evaluate model discrimination ability across thresholds

---

### 4. **Precision-Recall Curve** (04_precision_recall_curve.png)
- Trade-off between precision and recall
- Area Under Curve (AUC) score
- Optimal threshold identification

**Use Case**: Important for imbalanced datasets

---

### 5. **Class Distribution** (05_class_distribution.png)
Bar charts for:
- Training set: Normal vs Glaucoma ratio
- Validation set: Class balance
- Weighted sampling indicators

**Use Case**: Verify dataset composition and balance

---

### 6. **Metrics Comparison** (06_metrics_comparison.png)
7-metric comparison bar chart:
- Accuracy
- Precision
- Recall
- F1-Score
- MCC (Matthews Correlation Coefficient)
- Dice Score
- AUC-ROC

**Use Case**: Comprehensive performance overview

---

### 7. **Segmentation Results** (07_segmentation_results.png)
8-sample comparison showing:
- Column 1: Original fundus image
- Column 2: Ground truth optic disc/cup mask
- Column 3: Model-predicted segmentation mask with Dice score

**Use Case**: Visual validation of segmentation quality

---

### 8. **Segmentation Overlay** (08_segmentation_overlay.png)
6 samples with:
- Original image with predicted segmentation mask overlaid in red
- Class label (Normal/Glaucoma)
- Visual inspection of prediction accuracy

**Use Case**: Interpretability and qualitative assessment

---

### 9. **Grad-CAM Visualization** (09_gradcam_visualization.png)
3-column visualization per sample:
- Original image
- Grad-CAM heatmap (attention regions)
- Overlaid heatmap on original image

**Purpose**: Shows which image regions the model focuses on for classification

**Use Case**: Model interpretability and trust validation

---

### 10. **Saliency Maps** (10_saliency_maps.png)
3-column per sample showing:
- Original image
- Saliency map (gradient-based pixel importance)
- Overlay visualization

**Purpose**: Identifies most important pixels for classification

**Use Case**: Feature importance analysis

---

### 11. **Error Analysis** (11_error_analysis.png)
6-panel grid showing:
- Top row: False Positives (Normal misclassified as Glaucoma)
- Bottom row: False Negatives (Glaucoma misclassified as Normal)

**Purpose**: Debug model errors and identify edge cases

**Use Case**: Model improvement and refinement

---

### 12. **Capsule Activations** (12_capsule_activations.png)
4-sample visualization with:
- Original fundus image
- Capsule activation norms (Normal vs Glaucoma)
- Primary capsule activation heatmap

**Purpose**: Visualize learned hierarchical features

**Use Case**: Understand capsule network learned representations

---

### 13. **Dice Score Distribution** (13_dice_distribution.png)
2-panel visualization:
- Box plot: Dice score distribution by class
- Histogram: Frequency distribution of Dice scores

**Purpose**: Segmentation quality assessment

**Use Case**: Identify systematic segmentation errors

---

### 14. **Feature Maps** (14_feature_maps_sample_*.png)
4×4 grid showing:
- 16 feature channels from final encoder layer
- Color-coded activation patterns
- Per-channel visualization

**Purpose**: Understand CNN feature learning

**Use Case**: Debug feature extraction quality

---

### 15. **Metrics Table** (15_metrics_table.png)
Publication-ready summary table:

| Metric | Value |
|--------|-------|
| Accuracy | 0.XXXX |
| Precision | 0.XXXX |
| Recall | 0.XXXX |
| F1-Score | 0.XXXX |
| MCC | 0.XXXX |
| Dice Score | 0.XXXX |
| AUC-ROC | 0.XXXX |

**Use Case**: Paper/report inclusion

---

### 16. **Training Time Analysis** (16_training_time_analysis.png)
2-panel graph:
- Left: Time per epoch with mean reference line
- Right: Cumulative training time progression

**Purpose**: Performance optimization insights

**Use Case**: Hardware efficiency evaluation

---

### 17. **All Metrics Progression** (17_all_metrics_progression.png)
4-panel comprehensive view:
- Accuracy & F1-Score progression
- Precision & Recall progression
- MCC & Dice Score progression
- Loss components (Total, Seg, Cls)

**Purpose**: Complete training dynamics overview

**Use Case**: Publication-quality comprehensive results figure

---

## 🖼️ Sample Images

### Normal Eye Fundus Images
Located in: `images/normal/`

**Characteristics:**
- Clear optic disc boundary
- Cup-to-Disc Ratio (CDR) < 0.6
- Normal blood vessel configuration
- Healthy retinal appearance

**Example Files:**
```
normal_001.jpg
normal_002.jpg
normal_003.jpg
...
```

### Glaucoma Eye Fundus Images
Located in: `images/glaucoma/`

**Characteristics:**
- Enlarged optic cup
- Cup-to-Disc Ratio (CDR) > 0.6
- Vessel displacement and narrowing
- Possible neuroretinal rim thinning

**Example Files:**
```
glaucoma_001.jpg
glaucoma_002.jpg
glaucoma_003.jpg
...
```

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (for GPU acceleration)
```

### Dependencies
```bash
pip install torch torchvision
pip install pandas numpy
pip install scikit-learn opencv-python
pip install matplotlib seaborn
pip install tqdm pillow
```

### Setup
```bash
git clone https://github.com/Sowmyamaakam/Glaucoma_Detection.git
cd Glaucoma_Detection
```

---

## 📝 Usage

### Prepare Dataset
1. Place glaucoma images in `images/glaucoma/`
2. Place normal images in `images/normal/`
3. Create CSV with columns: `image_path`, `mask_path`, `label`, `split`

### Run Training
```bash
python glaucoma_code.py
```

### Model Output
The script generates:
- **Trained Models**: `best_model.pth`, `final_model.pth`
- **Visualizations**: 17 PNG files in `visualizations1/`
- **Metrics**: `test_metrics.csv`, `training_history.csv`

---

## 📈 Results

### Performance Metrics
- **Accuracy**: High classification accuracy
- **Sensitivity (Recall)**: Minimizes missed glaucoma cases
- **Specificity**: Reduces false alarms
- **Dice Score**: Good segmentation of optic disc/cup
- **AUC-ROC**: Strong discrimination between classes

### Key Findings
1. ✅ Capsule networks capture hierarchical optic disc features
2. ✅ Multi-task learning improves feature discrimination
3. ✅ Focal loss effectively handles class imbalance
4. ✅ Attention mechanisms provide clinical interpretability
5. ✅ Grad-CAM visualizations align with medical knowledge

---

## 🎯 Key Features

### 1. **Hybrid Architecture**
   - Combines CNNs (ResNet18) with Capsule Networks
   - Dual-task learning: Classification + Segmentation
   - Attention-based feature learning

### 2. **Advanced Loss Functions**
   - Focal Loss for classification (handles class imbalance)
   - BCE with Logits for segmentation
   - Weighted combination: 0.5×SegLoss + 2.0×ClsLoss

### 3. **Data Preprocessing**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Color Jitter augmentation
   - Normalization with ImageNet statistics

### 4. **Interpretability**
   - Grad-CAM attention maps
   - Saliency maps for pixel importance
   - Feature map visualization
   - Error analysis with misclassified samples

### 5. **Robust Evaluation**
   - Comprehensive metrics (7 different measures)
   - Confusion matrix analysis
   - ROC and Precision-Recall curves
   - Class-wise performance breakdown

### 6. **Medical Compliance**
   - 300 DPI output for publication
   - Clinical-grade visualization
   - Explainable AI for clinicians
   - Clear segmentation overlays

---

## 🔬 Technical Details

### Model Configuration
```python
CFG.batch_size = 8
CFG.lr = 1e-4
CFG.num_epochs = 50
CFG.patience = 2
CFG.img_size = 512
```

### Capsule Network Configuration
```python
Primary Capsules: 8 capsules, 16-dimensional
Routing Iterations: 3
Dropout: 0.3
Weight Initialization: Normal distribution (σ=0.01)
```

### Data Split
- Training: 70%
- Validation: 20%
- Test: 10%

---

## 📚 Research Paper References

This implementation is based on:
1. Capsule Networks (Hinton et al., 2017)
2. Dynamic Routing by Agreement (Sabour et al., 2017)
3. Grad-CAM for Visual Explanations (Selvaraju et al., 2016)
4. Focal Loss for Dense Object Detection (Lin et al., 2017)
5. CLAHE Medical Image Enhancement (Pizer et al., 1987)

---

## 💡 Future Improvements

- [ ] Ensemble methods (multi-model fusion)
- [ ] 3D volumetric analysis with OCT images
- [ ] Uncertainty quantification
- [ ] Edge deployment on mobile devices
- [ ] Real-time inference optimization
- [ ] Multi-site validation studies

---

## 📧 Contact

For questions or collaborations:
- GitHub: [@Sowmyamaakam](https://github.com/Sowmyamaakam)
- Project: Glaucoma Detection using HACapsNet

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- REFUGE Dataset
- PyTorch Community
- Medical Imaging Researchers
- Deep Learning Practitioners


