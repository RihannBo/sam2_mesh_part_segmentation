# Resume Training Instructions

## Overview
Resume training from **epoch 23 (best checkpoint)** with **reduced learning rates** (50% of original) for stable convergence.

## What Has Been Changed

### 1. New Resume Config Created
- **File**: `/home/mengnan/seg3d/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune_resume.yaml`
- **Changes**:
  - `base_lr: 5.0e-6` (reduced from 1.0e-5, 50% reduction)
  - `vision_lr: 2.5e-6` (reduced from 5.0e-6, 50% reduction)
  - `num_epochs: 100` (increased from 80 to allow more fine-tuning)
  - `warmup_steps: 200` (reduced from 1000, shorter warmup for resume)
  - Warmup scheduler lengths: `[0.0125, 0.9875]` (reduced warmup fraction)

### 2. Training Script Updated
- **File**: `/home/mengnan/seg3d/src/seg3d/training/run_trainer.py`
  - Added command-line argument support for config path and resume path
  - Auto-detects `best.pt` when using resume config

### 3. Training Loop Updated
- **File**: `/home/mengnan/seg3d/src/seg3d/training/train.py`
  - Updated resume logic to skip scheduler state loading
  - Scheduler starts fresh with new LR values from config
  - Scheduler calculates LR based on `where = start_epoch / num_epochs`

### 4. Resume Script Created
- **File**: `/home/mengnan/seg3d/resume_training.sh`
  - Automatically resumes from `best.pt` using resume config
  - Runs in detached screen session

## How to Resume Training

### Step 1: Stop Current Training (if still running)

```bash
# Option 1: Graceful stop (attach and interrupt)
screen -r training
# Press Ctrl+C twice, then Ctrl+A then D to detach

# Option 2: Kill screen session directly
screen -X -S training quit

# Option 3: Kill all training processes
pkill -f "python -m seg3d.training.run_trainer"
```

### Step 2: Verify Best Checkpoint Exists

```bash
ls -lh /home/mengnan/seg3d/checkpoints/best.pt
# Should show: best.pt from epoch 23 with val_loss=8.66
```

### Step 3: Resume Training

```bash
# Simply run the resume script
bash /home/mengnan/seg3d/resume_training.sh
```

Or manually:

```bash
cd /home/mengnan/seg3d
export PYTHONPATH=/home/mengnan/seg3d/src:$PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seg3d_env

python -m seg3d.training.run_trainer \
  /home/mengnan/seg3d/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune_resume.yaml \
  /home/mengnan/seg3d/checkpoints/best.pt
```

### Step 4: Monitor Training

```bash
# Attach to screen session
screen -r training_resume

# Monitor log file
tail -f /home/mengnan/seg3d/checkpoints/log.csv

# Detach from screen: Ctrl+A then D
```

## Expected Behavior

1. **Model**: Loads weights from epoch 23 (best checkpoint, val_loss=8.66)
2. **Optimizer**: Loads optimizer state (momentum, etc.) from checkpoint
3. **Scheduler**: Starts fresh with new LR schedule:
   - Base LR: 5.0e-6 (was 1.0e-5)
   - Vision LR: 2.5e-6 (was 5.0e-6)
   - Scheduler position: where = 23/100 = 0.23 (23% into training)
   - At this position, LR will be calculated from cosine schedule
4. **Training**: Continues from epoch 24 with reduced LR for stable convergence

## Expected Improvements

- ✅ **Stable convergence**: 50% lower LR should reduce oscillations
- ✅ **Progressive improvement**: May find better solutions with lower LR
- ✅ **Continued training**: Up to 100 epochs total (epochs 24-100 with new LR)

## Monitoring

Watch for:
- **Validation loss**: Should be more stable (fewer spikes)
- **Training loss**: Should continue decreasing smoothly
- **LR values**: Should follow cosine decay from new base values
- **Best checkpoint**: Should improve or remain stable (epoch 23 val_loss=8.66)

## If Training Fails

1. Check that `best.pt` exists and is valid
2. Verify resume config path is correct
3. Check CUDA memory (may need to reduce batch size)
4. Review logs: `tail -f /home/mengnan/seg3d/checkpoints/log.csv`

## Files Modified

- ✅ `/home/mengnan/seg3d/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune_resume.yaml` (new)
- ✅ `/home/mengnan/seg3d/src/seg3d/training/run_trainer.py` (updated)
- ✅ `/home/mengnan/seg3d/src/seg3d/training/train.py` (updated)
- ✅ `/home/mengnan/seg3d/resume_training.sh` (new)

