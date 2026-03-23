# seg3d/training/train.py

import os
import time
import sys
import warnings
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Suppress PyTorch deprecation warnings that interfere with progress bar
warnings.filterwarnings('ignore', category=FutureWarning)

from seg3d.training.utils.logging import CSVLogger, print_parameter_summary
from seg3d.training.utils.checkpointing import save_checkpoint, load_checkpoint

def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    device,
    ckpt_dir,
    num_epochs=80,
    base_lr=1e-4,
    vision_lr=None,  # Optional: separate LR for vision encoder (image_encoder.*)
    weight_decay=1e-4,
    accumulation_steps=4,
    optimizer=None,
    scheduler=None,
    scaler=None,
    use_amp=True,
    amp_dtype=None,
    resume_path=None,
    early_stop: bool = False,              # ← NEW
    early_stop_patience: int = 10,          # ← NEW
):
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------------------------------------
    # Parameter Summary (verify trainable/frozen)
    # -------------------------------------------------
    print_parameter_summary(model, verbose=True)

    # -------------------------------------------------
    # Optimizer / Scheduler / AMP
    # -------------------------------------------------
    optimizer_wrapper = None
    if optimizer is None:
        base_optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
        )
        if scheduler is None:
            scheduler = StepLR(
                base_optimizer,
                step_size=500,
                gamma=0.2,
            )
    else:
        optimizer_wrapper = optimizer if hasattr(optimizer, "step_schedulers") else None
        base_optimizer = (
            optimizer.optimizer if optimizer_wrapper is not None else optimizer
        )

    if scaler is None:
        scaler_enabled = use_amp and (amp_dtype is None or amp_dtype == torch.float16)
        scaler = GradScaler(enabled=scaler_enabled)

    # -------------------------------------------------
    # Resume (optional)
    # -------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0                  # ← NEW

    if resume_path is not None:
        print(f"🔁 Resuming from checkpoint: {resume_path}")
        
        # When resuming, we want to update LR from config if different
        # So load checkpoint but skip scheduler (let it start fresh with new LR from config)
        from seg3d.training.utils.checkpointing import load_checkpoint
        resume_info = load_checkpoint(
            resume_path,
            model,
            optimizer=base_optimizer,
            scheduler=None,  # Don't load scheduler - let it start fresh with new LR from config
            scaler=scaler,
            map_location=device,
        )
        start_epoch = resume_info["epoch"] + 1
        best_val_loss = resume_info["best_val_loss"]
        epochs_no_improve = resume_info.get("epochs_no_improve", 0)
        print(f"   Resumed from epoch {resume_info['epoch']}, best_val_loss: {best_val_loss:.4f}, epochs_no_improve: {epochs_no_improve}")
        
        # Note: LR will be set by scheduler on first training step based on 'where' = start_epoch/num_epochs
        # The scheduler uses new LR values from config (base_lr, vision_lr) and calculates LR
        # based on schedule position at start_epoch. So no manual LR update needed.
        print(f"   ℹ️  Scheduler will start fresh with new LR schedule from config")
        if vision_lr is not None:
            print(f"      (base_lr={base_lr:.2e}, vision_lr={vision_lr:.2e})")
        else:
            print(f"      (base_lr={base_lr:.2e}, vision_lr=None - using base_lr for all params)")
        print(f"      LR will be calculated based on schedule position at epoch {start_epoch}/{num_epochs}")

    # -------------------------------------------------
    # Logger
    # -------------------------------------------------
    # If resuming, truncate log at start_epoch to avoid duplicate entries
    resume_epoch = start_epoch if resume_path is not None else None
    logger = CSVLogger(ckpt_dir, resume_from_epoch=resume_epoch)

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Clear CUDA cache at start of each epoch to reduce fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # =========================
        # Train
        # =========================
        model.train()
        train_loss_sum = 0.0
        num_train_steps = 0

        base_optimizer.zero_grad(set_to_none=True)

        # Use tqdm with proper settings to avoid multiple progress bars
        # Suppress all warnings during training to avoid interfering with progress bar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Check if we're in a TTY (interactive terminal)
            is_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
            
            # Set environment variable to suppress PyTorch warnings
            os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
            
            pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch} [TRAIN]", 
                leave=False, 
                ncols=120,  # Fixed width
                mininterval=2.0,  # Update at most every 2 seconds to reduce clutter and avoid multiple bars
                file=sys.stderr,  # Always use stderr
                dynamic_ncols=False,  # Fixed width
                disable=False,
                ascii=False,  # Use Unicode characters
                smoothing=0.1,  # Smooth the rate estimate
            )

        for step, batch in enumerate(pbar):
            batch = batch.to(device)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                outputs = model(batch)
                # Break references to heavy auxiliary tensors that are not used by the loss.
                # This helps reduce autograd graph retention and peak memory.
                if isinstance(outputs, list):
                    for frame_out in outputs:
                        if not isinstance(frame_out, dict):
                            continue
                        if frame_out.get("obj_ptr", None) is not None:
                            frame_out["obj_ptr"] = frame_out["obj_ptr"].detach()
                        if frame_out.get("maskmem_features", None) is not None:
                            frame_out["maskmem_features"] = frame_out[
                                "maskmem_features"
                            ].detach()
                        if frame_out.get("maskmem_pos_enc", None) is not None:
                            mpe = frame_out["maskmem_pos_enc"]
                            if isinstance(mpe, list):
                                frame_out["maskmem_pos_enc"] = [
                                    t.detach() if torch.is_tensor(t) else t for t in mpe
                                ]
                            elif torch.is_tensor(mpe):
                                frame_out["maskmem_pos_enc"] = mpe.detach()

                        for k in [
                            "multistep_pred_masks",
                            "multistep_pred_masks_high_res",
                            "multistep_pred_multimasks",
                            "multistep_pred_multimasks_high_res",
                        ]:
                            # NOTE: `training.loss_fns.MultiStepMultiMasksAndIous` uses
                            # `multistep_pred_multimasks_high_res` and `multistep_pred_ious`,
                            # so we must NOT detach those if present.
                            if k == "multistep_pred_multimasks_high_res":
                                continue
                            v = frame_out.get(k, None)
                            if v is None:
                                continue
                            if torch.is_tensor(v):
                                frame_out[k] = v.detach()
                            elif isinstance(v, list):
                                frame_out[k] = [
                                    t.detach() if torch.is_tensor(t) else t for t in v
                                ]
                loss = loss_fn(outputs, batch.masks)

                if isinstance(loss, dict):
                    loss = loss["core_loss"]

                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if optimizer_wrapper is not None:
                exact_epoch = epoch + float(step) / max(1, len(train_loader))
                where = float(exact_epoch) / max(1, num_epochs)
                if where <= 1.0 + 1e-8:
                    optimizer_wrapper.step_schedulers(
                        where, step=int(exact_epoch * max(1, len(train_loader)))
                    )

            if (step + 1) % accumulation_steps == 0:
                scaler.step(base_optimizer)
                scaler.update()
                base_optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and optimizer_wrapper is None:
                    scheduler.step()
                # Clear cache periodically to reduce fragmentation
                if torch.cuda.is_available() and (step + 1) % (accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()

            train_loss_sum += loss.item() * accumulation_steps
            num_train_steps += 1

            pbar.set_postfix(
                loss=train_loss_sum / max(1, num_train_steps),
                lr=base_optimizer.param_groups[0]["lr"],
            )

        train_loss = train_loss_sum / max(1, num_train_steps)

        # =========================
        # Validation
        # =========================
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        num_val_steps = 0

        with torch.no_grad():
            # Use tqdm with proper settings to avoid multiple progress bars
            # Check if we're in a TTY (interactive terminal)
            is_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
            
            pbar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch} [VAL]", 
                leave=False, 
                ncols=120 if is_tty else None,  # Fixed width only for TTY
                mininterval=0.5,  # Update at most every 0.5 seconds
                file=sys.stderr,  # Always use stderr
                dynamic_ncols=not is_tty,  # Dynamic width when not TTY
                disable=False,
                ascii=False,  # Use Unicode characters
                smoothing=0.1,  # Smooth the rate estimate
            )

            for batch in pbar:
                batch = batch.to(device)

                with autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = model(batch)
                    loss = loss_fn(outputs, batch.masks)

                if isinstance(loss, dict):
                    val_loss = loss["core_loss"].item()
                    val_iou = loss.get("loss_iou", torch.tensor(0.0)).item()
                else:
                    val_loss = loss.item()
                    val_iou = 0.0

                val_loss_sum += val_loss
                val_iou_sum += val_iou
                num_val_steps += 1

                pbar.set_postfix(
                    val_loss=val_loss_sum / max(1, num_val_steps)
                )

        val_loss = val_loss_sum / max(1, num_val_steps)
        val_iou = val_iou_sum / max(1, num_val_steps)

        # =========================
        # Checkpointing
        # =========================
        save_checkpoint(
            ckpt_dir,
            "last.pt",
            model,
            base_optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            epochs_no_improve,
        )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                ckpt_dir,
                "best.pt",
                model,
                base_optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                epochs_no_improve,
            )
        else:
            epochs_no_improve += 1

        # =========================
        # Logging
        # =========================
        epoch_time = time.time() - epoch_start

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_iou=val_iou,
            lr=base_optimizer.param_groups[0]["lr"],
            epoch_time_sec=epoch_time,
            best_val_loss=best_val_loss,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train: {train_loss:.4f} | "
            f"val: {val_loss:.4f} | "
            f"best: {best_val_loss:.4f}"
        )

        # =========================
        # Early stopping (optional)
        # =========================
        if early_stop and epochs_no_improve >= early_stop_patience:
            print(
                f"🛑 Early stopping triggered after "
                f"{epochs_no_improve} epochs without improvement."
            )
            break

    logger.close()
