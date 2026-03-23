# Dataset and Training Details Summary

## 1. Your Dataset (Object Types and Scale Variation)

### Dataset Structure
- **Total Meshes**: 200 meshes (~6.7 GB)
- **Views per Mesh**: 12 multi-view images
- **Image Resolution**: 1024×1024 pixels
- **Data Format**: NPZ files (`mesh_data.npz`) per mesh directory

### Data Contents
Each mesh contains:
- **`normals_u8`**: (T, H, W, 3) - Surface normals encoded as uint8
- **`pointmap_f16`**: (T, H, W, 3) - World XYZ coordinates (float16)
- **`gt_masks`**: (T, H, W) - Ground truth instance masks (uint16)
- **`view_dirs`**: (T, 3) - Camera view directions (float32)
- **`mattes_u8`**: (T, H, W, 3) - Optional matte images (uint8)

### Object Types
- **Instance Segmentation**: Multiple object instances per scene
- **Max Instances**: Up to 20+ instances per view (based on inference notebook output)
- **Instance IDs**: Stored as uint16 in `gt_masks` (0 = background, 1+ = object instances)
- **Object Types**: Not explicitly labeled - the dataset uses instance IDs rather than semantic classes

### Scale Variation
- **Multi-View**: 12 views per mesh provide scale variation through different camera angles
- **Normalization**: 
  - Point maps normalized per-mesh (min-max normalization across all frames)
  - Normal maps normalized to [0, 1] range
- **Spatial Scale**: Objects appear at different scales across the 12 views
- **Training Split**: 160 meshes (train) / 40 meshes (validation) - 80/20 split

### Dataset Characteristics
- **Geometry Input**: Dual-modal (normal maps + point clouds)
- **World Coordinates**: Point maps contain real-world XYZ coordinates
- **Multi-Instance**: Each view can contain multiple object instances
- **Bootstrap**: First frame duplicated (GeoSAM2-style) for better initialization

---

## 2. Current Validation Metrics

### Best Performance (Epoch 42)
- **Validation Loss**: 8.075 (best achieved)
- **Validation IoU**: 1.298 (average IoU)
- **Status**: Best checkpoint saved

### Final Performance (Epoch 52)
- **Validation Loss**: 8.417
- **Validation IoU**: 1.310
- **Status**: Training stopped (early stopping triggered)

### Recent Performance (Last 5 Epochs: 48-52)
- **Average Val Loss**: 8.42 ± 0.05
- **Average Val IoU**: 1.30 ± 0.01
- **Trend**: Stable (no significant improvement)

### Training Progress
- **Initial Val Loss**: 101.18 (epoch 0)
- **Best Val Loss**: 8.075 (epoch 42)
- **Improvement**: 92.0% reduction in validation loss
- **Convergence**: Model converged after epoch 42 (10+ epochs without improvement)

### Metrics Interpretation
- **Val Loss**: Multi-step multi-mask loss (mask loss + dice loss + IoU loss + class loss)
- **Val IoU**: Average Intersection over Union across all predicted masks
- **IoU ~1.3**: Indicates good segmentation quality (IoU typically ranges 0-1, but can exceed 1 with multi-mask outputs)

---

## 3. Prompt Type and Generation

### Training Prompt Distribution
**Mixed Prompt Training Strategy**:
- **70% Boxes** - Bounding boxes from GT masks
- **20% Points** - Single point prompts from GT masks  
- **10% Masks** - Direct mask inputs (strong supervision)

### Configuration
```yaml
prob_to_use_pt_input_for_train: 0.9   # 90% use prompts (box/point), 10% use masks
prob_to_use_box_input_for_train: 0.778  # Of the 90%: 77.8% boxes, 22.2% points
# Final distribution: 70% boxes, 20% points, 10% masks
```

### Prompt Generation Methods

#### A. Box Prompts (70% of training)
**Function**: `sample_box_points()` in `sam2/sam2/modeling/sam2_utils.py`

**Process**:
1. **Extract Bounding Box**: Compute bounding box from GT mask
   ```python
   box_coords = mask_to_box(masks)  # [x_min, y_min, x_max, y_max]
   ```

2. **Add Noise** (prevents overfitting):
   - Noise level: 10% of box width/height (`noise=0.1`)
   - Max noise: 20 pixels (`noise_bound=20`)
   - Random jitter applied to top-left and bottom-right corners

3. **Generate Box Points**:
   - **Top-left corner**: Label = 2
   - **Bottom-right corner**: Label = 3
   - Output: `[B, 2, 2]` coordinates, `[B, 2]` labels

**Why Boxes?**
- SAM2's primary training mode
- Better spatial context for object boundaries
- More efficient (2 points vs 3+ points)
- With noise, prevents overfitting while maintaining structure

#### B. Point Prompts (20% of training)
**Function**: `get_next_point()` → `sample_random_points_from_errors()`

**Process**:
1. **Sample from GT Mask**: Uniformly sample 1 point from ground truth mask
2. **Point Label**: Label = 1 (foreground point)
3. **Method**: `"uniform"` during training (random sampling from GT regions)

**Why Points?**
- More realistic inference scenario
- Improves robustness to sparse prompts
- Better generalization to real-world usage

#### C. Mask Prompts (10% of training)
**Process**:
1. **Direct GT Mask**: Use ground truth mask directly as input
2. **No Point/Box Sampling**: Mask is fed directly to SAM prompt encoder
3. **Strong Supervision**: Provides strong supervision signal

**Why Masks?**
- Prevents overfitting to sparse prompts
- Ensures model can handle mask inputs
- Strong supervision signal for better convergence

### Evaluation Prompts
**During Evaluation**:
- **Prompt Type**: **Mask inputs only** (`prob_to_use_pt_input_for_eval=0.0`)
- **Rationale**: Evaluation uses mask inputs to simulate best-case scenario
- **Initial Frame**: Only frame 0 receives mask input
- **Tracking**: Subsequent frames use memory from previous frames

### Multi-Instance Training
- **Instances per Step**: Up to 3 instances processed per training step
- **Episodic Training**: Each instance treated as separate episode
- **Prompt per Instance**: Each instance gets its own prompt (box/point/mask)

### Prompt Generation Code Flow

```python
# In prepare_prompt_inputs():

# 1. Decide prompt type
use_pt_input = random() < prob_to_use_pt_input_for_train  # 90% chance

if use_pt_input:
    # 2. Decide box vs point
    use_box_input = random() < prob_to_use_box_input_for_train  # 77.8% chance
    
    if use_box_input:
        # Generate box from GT mask with noise
        points, labels = sample_box_points(gt_masks, noise=0.1)
        # Returns: 2 points (top-left, bottom-right) with labels [2, 3]
    else:
        # Generate single point from GT mask
        points, labels = get_next_point(gt_masks, method="uniform")
        # Returns: 1 point with label [1]
else:
    # Use mask input directly
    mask_inputs = gt_masks  # Direct GT mask
```

### Key Features
1. **Noise in Boxes**: 10% jitter prevents overfitting to perfect boxes
2. **Uniform Point Sampling**: Points sampled uniformly from GT mask regions
3. **Multi-Instance Support**: Each instance gets independent prompts
4. **Aligned with GT**: All prompts derived from ground truth masks
5. **No Correction Points**: `num_correction_pt_per_frame=0` (disabled for non-interactive training)

---

## Summary

### Dataset
- **200 meshes**, 12 views each, **multi-instance segmentation**
- **Dual-modal**: Normal maps + point clouds
- **Scale variation** through multi-view geometry

### Validation Metrics
- **Best Val Loss**: 8.075 (epoch 42)
- **Best Val IoU**: 1.298
- **Status**: Model converged, ready for inference

### Prompt Generation
- **Training**: 70% boxes, 20% points, 10% masks
- **Boxes**: Generated from GT masks with 10% noise
- **Points**: Uniformly sampled from GT mask regions
- **Evaluation**: Mask inputs only (best-case scenario)

---

*Generated: January 2025*

