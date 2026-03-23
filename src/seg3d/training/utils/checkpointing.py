# seg3d/training/checkpointing.py

import os
import torch


def save_checkpoint(
    ckpt_dir: str,
    name: str,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
    epochs_no_improve: int = 0,
):
    """
    Saves a full training checkpoint.
    
    Args:
        ckpt_dir: Directory to save checkpoint
        name: Checkpoint filename
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Optional scheduler to save
        scaler: Optional AMP scaler to save
        epoch: Current epoch number
        best_val_loss: Best validation loss so far
        epochs_no_improve: Number of epochs without improvement (for early stopping)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
    }

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    torch.save(ckpt, path)
    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """
    Loads a training checkpoint and restores states.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        scaler: Optional AMP scaler to load state into
        map_location: Device to load checkpoint to

    Returns:
        dict with keys:
            - epoch: Epoch number (or -1 if not found)
            - best_val_loss: Best validation loss (or inf if not found)
            - epochs_no_improve: Epochs without improvement (or 0 if not found)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"📂 Loading checkpoint from: {path}")
    ckpt = torch.load(path, map_location=map_location)

    # Validate checkpoint structure
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' key. Available keys: {list(ckpt.keys())}")

    # Model
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
    if missing_keys:
        print(f"⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"     - {key}")
        else:
            for key in missing_keys[:5]:
                print(f"     - {key}")
            print(f"     ... and {len(missing_keys) - 5} more")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"     - {key}")
        else:
            for key in unexpected_keys[:5]:
                print(f"     - {key}")
            print(f"     ... and {len(unexpected_keys) - 5} more")

    # Optimizer
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("✓ Optimizer state loaded")
        except Exception as e:
            print(f"⚠️  Failed to load optimizer state: {e}")

    # Scheduler
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("✓ Scheduler state loaded")
        except Exception as e:
            print(f"⚠️  Failed to load scheduler state: {e}")

    # AMP scaler
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print("✓ AMP scaler state loaded")
        except Exception as e:
            print(f"⚠️  Failed to load scaler state: {e}")

    return {
        "epoch": ckpt.get("epoch", -1),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "epochs_no_improve": ckpt.get("epochs_no_improve", 0),
    }


def load_best_model_for_inference(
    cfg_path: str,
    ckpt_path: str | None = None,
    device: str | torch.device | None = None,
):
    """
    Instantiate the training model class and load best.pt (or a provided checkpoint).
    Returns (model, checkpoint_metadata).
    """
    try:
        from training.utils.train_utils import register_omegaconf_resolvers
        register_omegaconf_resolvers()
    except Exception:
        pass

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.trainer.checkpoint.save_dir, "best.pt")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = instantiate(cfg.trainer.model)
    checkpoint_info = load_checkpoint(
        ckpt_path, model, map_location="cpu"
    )
    model = model.to(device)
    model.eval()
    return model, checkpoint_info
