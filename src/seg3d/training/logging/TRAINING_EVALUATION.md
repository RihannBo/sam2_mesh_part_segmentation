# Training Evaluation Analysis: seg3d

## Overview
This document analyzes how the `seg3d` training code evaluates models during training, including metrics, evaluation frequency, and checkpointing strategies.

---

## 1. Evaluation Architecture

### Two Training Implementations

The codebase has **two training implementations**:

1. **`multiview_trainer.py`**: Full-featured trainer with meters, distributed training, and comprehensive logging
2. **`train.py`**: Simpler training loop with basic validation

---

## 2. Evaluation Process

### A. Validation Epoch (`multiview_trainer.py`)

**Location**: `src/seg3d/training/multiview_trainer.py::val_epoch()`

**Evaluation Flow**:

```python
def val_epoch(self, val_loader, phase):
    # 1. Set model to eval mode
    model.eval()
    
    # 2. Initialize loss meters
    loss_mts = OrderedDict([...])  # Tracks loss per batch
    
    # 3. Run validation loop
    for batch in val_loader:
        with torch.no_grad():  # No gradients during eval
            with torch.cuda.amp.autocast():  # Mixed precision
                # Forward pass
                loss_dict, batch_size, extra_losses = self._step(
                    batch, model, phase
                )
                
                # Update loss meters
                loss_mts[loss_key].update(loss.item(), batch_size)
    
    # 4. Compute and log metrics
    out_dict = self._log_meters_and_save_best_ckpts(phases)
    
    return out_dict  # Contains all metrics
```

**Key Features**:
- **No gradients**: `torch.no_grad()` during evaluation
- **Mixed precision**: Uses AMP (Automatic Mixed Precision) if enabled
- **Loss tracking**: Tracks loss per batch and averages
- **Meter synchronization**: Meters are synchronized across GPUs in distributed training

---

### B. Simple Validation (`train.py`)

**Location**: `src/seg3d/training/train.py`

**Evaluation Flow**:

```python
# Validation
model.eval()
val_loss_sum = 0.0
val_iou_sum = 0.0
num_val_steps = 0

with torch.no_grad():
    for batch in val_loader:
        with autocast():
            outputs = model(batch)
            loss = loss_fn(outputs, batch.masks)
        
        # Extract loss components
        if isinstance(loss, dict):
            val_loss = loss["core_loss"].item()
            val_iou = loss.get("loss_iou", torch.tensor(0.0)).item()
        else:
            val_loss = loss.item()
            val_iou = 0.0
        
        val_loss_sum += val_loss
        val_iou_sum += val_iou
        num_val_steps += 1

# Average metrics
val_loss = val_loss_sum / max(1, num_val_steps)
val_iou = val_iou_sum / max(1, num_val_steps)
```

**Metrics Computed**:
1. **Validation Loss** (`val_loss`): Core loss from loss function
2. **Validation IoU** (`val_iou`): IoU loss component (if available)

---

## 3. Metrics and Loss Functions

### Loss Function Structure

**Location**: `training/loss_fns.py` (from SAM2 codebase)

The loss function returns a dictionary with multiple components:

```python
loss = {
    "core_loss": <main_loss>,      # Used for backprop
    "loss_mask": <mask_loss>,      # Binary cross-entropy
    "loss_dice": <dice_loss>,       # Dice loss
    "loss_iou": <iou_loss>,         # IoU prediction loss
    "loss_class": <class_loss>,     # Classification loss (if applicable)
}
```

**Loss Components**:
1. **Mask Loss**: Binary cross-entropy between predicted and ground truth masks
2. **Dice Loss**: Dice coefficient loss for mask overlap
3. **IoU Loss**: IoU prediction loss (if IoU heads are trained)
4. **Class Loss**: Classification loss (if applicable)

### Metrics Tracked

**From `multiview_trainer.py`**:

1. **Loss Metrics**:
   - `Losses/val_{key}_loss`: Validation loss per dataset key
   - `Losses/train_{key}_loss`: Training loss per dataset key
   - Individual loss components (mask, dice, iou, class)

2. **Meters** (if configured):
   - Meters are instantiated from config (`meters_conf`)
   - They track metrics like IoU, accuracy, etc.
   - Meters have `update()` and `compute_synced()` methods
   - Examples: IoU meters, accuracy meters (from SAM2 training codebase)

**From `train.py`**:
- `train_loss`: Average training loss
- `val_loss`: Average validation loss  
- `val_iou`: Average validation IoU

---

## 4. Evaluation Frequency

### Validation Schedule

**`multiview_trainer.py`**:

```python
val_epoch_freq: int = 1  # Validate every N epochs

def is_intermediate_val_epoch(self, epoch):
    return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1
```

**Validation runs**:
- Every `val_epoch_freq` epochs (default: every epoch)
- After training loop completes
- Before resuming from checkpoint (if previous epoch was validation epoch)

**`train.py`**:
- Validates **every epoch** after training

---

## 5. Checkpointing Strategy

### Best Model Selection

**`multiview_trainer.py`**:

```python
def _log_meters_and_save_best_ckpts(self, phases):
    for key, meter in self._get_meters(phases).items():
        meter_output = meter.compute_synced()
        is_better_check = getattr(meter, "is_better", None)
        
        # Check if current metric is better than best
        if is_better_check(meter_value, self.best_meter_values[tracked_meter_key]):
            self.best_meter_values[tracked_meter_key] = meter_value
            
            # Save checkpoint if metric is in save_best_meters list
            if key in self.checkpoint_conf.save_best_meters:
                checkpoint_save_keys.append(tracked_meter_key)
    
    if len(checkpoint_save_keys) > 0:
        self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)
```

**Checkpoint Types**:
1. **Regular checkpoints**: Saved every `save_freq` epochs
2. **Best checkpoints**: Saved when metrics improve (based on `save_best_meters` config)
3. **Epoch checkpoints**: Saved at specific epochs (`save_list`)

**`train.py`**:

```python
improved = val_loss < best_val_loss

if improved:
    best_val_loss = val_loss
    save_checkpoint(ckpt_dir, "best.pt", ...)  # Save best model

# Always save last checkpoint
save_checkpoint(ckpt_dir, "last.pt", ...)
```

**Checkpoint Strategy**:
- **Best model**: Saved when `val_loss` improves
- **Last model**: Saved every epoch
- **Early stopping**: Stops if no improvement for `early_stop_patience` epochs

---

## 6. Evaluation Configuration

### Model Evaluation Settings

**Location**: `src/seg3d/training/model/multiview_sam2.py`

**Evaluation-specific parameters**:

```python
prob_to_use_pt_input_for_eval=0.0      # No point prompts during eval
prob_to_use_box_input_for_eval=0.0     # No box prompts during eval
num_frames_to_correct_for_eval=1       # Correction on first frame only
num_init_cond_frames_for_eval=1        # Use first frame as conditioning
rand_init_cond_frames_for_eval=False   # Deterministic conditioning
pt_sampling_for_eval="center"          # Point sampling method
forward_backbone_per_frame_for_eval=False  # Process all frames at once
```

**Key Differences: Train vs Eval**:

| Setting | Training | Evaluation |
|---------|----------|------------|
| Point prompts | `prob_to_use_pt_input_for_train` | `0.0` (disabled) |
| Box prompts | `prob_to_use_box_input_for_train` | `0.0` (disabled) |
| Correction frames | `num_frames_to_correct_for_train` | `1` (first frame only) |
| Conditioning frames | Random (if enabled) | Fixed (first frame) |
| Point sampling | `"uniform"` | `"center"` |
| Backbone processing | Per-frame (memory efficient) | All frames (faster) |

---

## 7. Logging and Monitoring

### Logged Metrics

**`multiview_trainer.py`**:

**JSON Logs**:
- `train_stats.json`: Training metrics per epoch
- `val_stats.json`: Validation metrics per epoch
- `best_stats.json`: Best metric values

**TensorBoard Logs**:
- Loss curves: `Losses/{phase}_{key}_loss`
- Step losses: `Step_Losses/{phase}_{key}_loss`
- Progress meters: `Step_Stats/{phase}/{meter_name}`
- Optimizer: `Optim/{param_group}/{option}`

**`train.py`**:

**CSV Logs**:
- `train_log.csv`: Epoch, train_loss, val_loss, val_iou, lr, epoch_time, best_val_loss

---

## 8. Evaluation Data Flow

### Complete Evaluation Pipeline

```
1. Data Loading
   └─ val_loader.get_loader(epoch)
      └─ Loads validation batches

2. Model Forward Pass
   └─ model.eval()  # Set to evaluation mode
   └─ with torch.no_grad():  # Disable gradients
      └─ outputs = model(batch)
         └─ MultiViewSAM2Train.forward()
            └─ Processes all frames
            └─ Returns list of frame outputs

3. Loss Computation
   └─ loss = loss_fn(outputs, batch.masks)
      └─ MultiStepMultiMasksAndIous.forward()
         └─ Computes: mask_loss, dice_loss, iou_loss
         └─ Returns: {"core_loss": ..., "loss_mask": ..., ...}

4. Metric Aggregation
   └─ loss_mts[loss_key].update(loss.item(), batch_size)
   └─ meters.update(find_stages=outputs, find_metadatas=batch.metadata)

5. Synchronization (Distributed)
   └─ meter.compute_synced()  # Average across GPUs

6. Logging & Checkpointing
   └─ Log metrics to TensorBoard/JSON
   └─ Save best checkpoints if metrics improved
```

---

## 9. Key Evaluation Features

### A. Distributed Evaluation

- **Synchronized metrics**: All meters are synchronized across GPUs
- **Barrier synchronization**: `dist.barrier()` every 10 iterations
- **Rank 0 logging**: Only rank 0 saves checkpoints and logs

### B. Memory Efficiency

- **No gradient computation**: `torch.no_grad()` saves memory
- **Detached tensors**: Heavy auxiliary tensors are detached during training
- **Per-frame backbone**: Option to process backbone per-frame during eval

### C. Reproducibility

- **Deterministic evaluation**: Fixed random seeds for evaluation
- **Fixed conditioning**: First frame always used as conditioning
- **Center point sampling**: Consistent point sampling method

---

## 10. Summary

### Evaluation Metrics

1. **Primary Metrics**:
   - Validation Loss (core_loss)
   - Validation IoU (if IoU heads trained)
   - Training Loss

2. **Secondary Metrics** (from meters):
   - IoU per class/object
   - Accuracy metrics
   - Other task-specific metrics

### Evaluation Frequency

- **Default**: Every epoch
- **Configurable**: Via `val_epoch_freq`

### Checkpointing

- **Best model**: Saved when validation loss/metrics improve
- **Regular checkpoints**: Saved at fixed intervals
- **Early stopping**: Available in `train.py`

### Key Files

- **Trainer**: `src/seg3d/training/multiview_trainer.py`
- **Simple trainer**: `src/seg3d/training/train.py`
- **Model**: `src/seg3d/training/model/multiview_sam2.py`
- **Loss**: `training/loss_fns.py` (from SAM2)

---

## Notes

1. **Meters are configurable**: The actual metrics computed depend on the `meters_conf` in the training config
2. **Loss function is from SAM2**: The loss implementation is from the SAM2 training codebase
3. **Evaluation is deterministic**: Evaluation uses fixed seeds and deterministic settings
4. **Distributed support**: Full support for multi-GPU evaluation with metric synchronization

