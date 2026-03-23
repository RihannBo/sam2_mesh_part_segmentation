# seg3d/training/run_train.py

import os
import torch
import fnmatch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from seg3d.training.train import train
from seg3d.training.utils.data_utils import collate_fn
from seg3d.training.utils.logging import print_parameter_summary
from training.optimizer import construct_optimizer
from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.train_utils import register_omegaconf_resolvers


def get_amp_type(amp_type: str = None):
    """Convert amp_dtype string to torch dtype."""
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
    if amp_type == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


class StripBaseLayerKernel:
    """
    Removes .base_layer from checkpoint keys.
    This is needed because CkptKeyMapperKernel adds .base_layer for LoRA,
    but GeoSAM2MultimodalEncoder uses hooks and keeps original qkv.weight keys.
    """
    def __call__(self, state_dict: dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            # Strip .base_layer from keys (e.g., qkv.base_layer.weight -> qkv.weight)
            new_key = key.replace(".base_layer.weight", ".weight")
            new_key = new_key.replace(".base_layer.bias", ".bias")
            new_state_dict[new_key] = value
        return new_state_dict


def _filter_patterns(patterns, keys):
    if not patterns:
        return []
    key_list = list(keys)
    kept = []
    for pattern in patterns:
        if fnmatch.filter(key_list, pattern):
            kept.append(pattern)
    return kept


def _validate_optimizer_params(optimizer, model, param_allowlist):
    if optimizer is None or not param_allowlist:
        return
    if hasattr(optimizer, "optimizer"):
        opt_param_groups = optimizer.optimizer.param_groups
    else:
        opt_param_groups = optimizer.param_groups
    opt_params = {p for group in opt_param_groups for p in group["params"]}
    expected_params = {
        p for name, p in model.named_parameters() if name in param_allowlist
    }
    if opt_params == expected_params:
        return
    missing_names = [
        name
        for name, p in model.named_parameters()
        if name in param_allowlist and p not in opt_params
    ]
    extra_names = [
        name
        for name, p in model.named_parameters()
        if name not in param_allowlist and p in opt_params
    ]
    raise RuntimeError(
        "Optimizer param groups do not match the requested trainable parameters. "
        f"Missing: {missing_names[:10]}{'...' if len(missing_names) > 10 else ''} "
        f"Extra: {extra_names[:10]}{'...' if len(extra_names) > 10 else ''}"
    )


def _ensure_interval_scaling(node):
    if node is None:
        return
    if isinstance(node, dict):
        target = node.get("_target_")
        if target == "fvcore.common.param_scheduler.CompositeParamScheduler":
            schedulers = node.get("schedulers", [])
            count = len(schedulers) if schedulers is not None else 0
            interval = node.get("interval_scaling", None)
            if count > 0:
                if interval is None or isinstance(interval, str):
                    node["interval_scaling"] = ["rescaled"] * count
                elif isinstance(interval, (list, tuple)) and len(interval) != count:
                    node["interval_scaling"] = list(interval)[:count] + [
                        "rescaled"
                    ] * max(0, count - len(interval))
        for value in node.values():
            _ensure_interval_scaling(value)
        return
    if isinstance(node, (list, tuple)):
        for value in node:
            _ensure_interval_scaling(value)
        return
    if OmegaConf.is_config(node):
        if OmegaConf.is_list(node):
            for value in node:
                _ensure_interval_scaling(value)
            return
        if OmegaConf.is_dict(node):
            if "_target_" in node and node.get(
                "_target_"
            ) == "fvcore.common.param_scheduler.CompositeParamScheduler":
                schedulers = node.get("schedulers", [])
                count = len(schedulers) if schedulers is not None else 0
                interval = node.get("interval_scaling", None)
                if count > 0:
                    if interval is None or isinstance(interval, str):
                        node["interval_scaling"] = ["rescaled"] * count
                    elif OmegaConf.is_list(interval) and len(interval) != count:
                        node["interval_scaling"] = list(interval)[:count] + [
                            "rescaled"
                        ] * max(0, count - len(interval))
            for _, value in node.items():
                _ensure_interval_scaling(value)
        return


def main(cfg_path: str, resume_path: str = None):
    """
    Main training function.
    
    Args:
        cfg_path: Path to training config YAML file
        resume_path: Optional path to checkpoint to resume from
    """
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 Cleared CUDA cache. Free memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)  # Resolve all interpolations (like ${times:...})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = instantiate(cfg.data.train)
    val_ds = instantiate(cfg.data.val)

    train_loader = train_ds.get_loader(collate_fn=collate_fn, dict_key="train")
    val_loader = val_ds.get_loader(collate_fn=collate_fn, dict_key="val")

    model = instantiate(cfg.trainer.model)

    if (
        "trainer" in cfg
        and "checkpoint" in cfg.trainer
        and "model_weight_initializer" in cfg.trainer.checkpoint
    ):
        initializer_cfg = cfg.trainer.checkpoint.model_weight_initializer
        
        # Add StripBaseLayerKernel to remove .base_layer that CkptKeyMapperKernel adds
        # CkptKeyMapperKernel adds .base_layer for LoRA, but GeoSAM2MultimodalEncoder
        # uses hooks and keeps original qkv.weight keys (no .base_layer)
        if "state_dict" in initializer_cfg and "checkpoint_kernels" in initializer_cfg.state_dict:
            # Add strip kernel to the kernel list after the mapping kernel
            kernel_list = list(initializer_cfg.state_dict.checkpoint_kernels)
            kernel_list.append({
                "_target_": "seg3d.training.run_trainer.StripBaseLayerKernel"
            })
            initializer_cfg.state_dict.checkpoint_kernels = kernel_list
        
        state_dict_cfg = initializer_cfg.get("state_dict", None)
        if state_dict_cfg is not None:
            state_dict = instantiate(state_dict_cfg)
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing_keys = model_keys - ckpt_keys
            unexpected_keys = ckpt_keys - model_keys
            ignore_missing = list(initializer_cfg.get("ignore_missing_keys", []) or [])
            ignore_unexpected = list(initializer_cfg.get("ignore_unexpected_keys", []) or [])
            ignore_missing = _filter_patterns(ignore_missing, missing_keys)
            ignore_unexpected = _filter_patterns(ignore_unexpected, unexpected_keys)
            model = load_state_dict_into_model(
                state_dict=state_dict,
                model=model,
                strict=bool(initializer_cfg.get("strict", True)),
                ignore_missing_keys=ignore_missing or None,
                ignore_unexpected_keys=ignore_unexpected or None,
            )
        else:
            model_weight_initializer = instantiate(initializer_cfg)
            if model_weight_initializer is not None:
                model = model_weight_initializer(model=model)

    model = model.to(device)
    
    # Print a concise parameter summary unless explicitly disabled.
    if os.environ.get("SEG3D_PRINT_PARAM_SUMMARY", "1") != "0":
        verbose_summary = os.environ.get("SEG3D_VERBOSE_PARAM_SUMMARY", "0") == "1"
        print_parameter_summary(model, verbose=verbose_summary)
    
    # Check if config uses a trainer class (like MultiViewSAM2Trainer)
    if "trainer" in cfg and "_target_" in cfg.trainer:
        # Use the trainer from config
        print("📦 Using trainer from config:", cfg.trainer._target_)
        
        # Set up distributed environment variables for single-GPU training
        # (The trainer will initialize distributed backend itself)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        # Ensure interval_scaling is set for CompositeParamScheduler
        if "optim" in cfg.trainer and "options" in cfg.trainer.optim:
            _ensure_interval_scaling(cfg.trainer.optim.options)
        
        # Set resume path in checkpoint config if provided
        if resume_path is not None:
            if "checkpoint" not in cfg.trainer:
                cfg.trainer["checkpoint"] = {}
            cfg.trainer.checkpoint["resume_from"] = resume_path
        
        # Create trainer config by copying and removing unsupported params
        # MultiViewSAM2Trainer doesn't accept early_stop_patience, num_instances_per_step, etc.
        trainer_cfg_dict = dict(cfg.trainer)
        unsupported_params = ["num_instances_per_step", "dict_key"]
        for param in unsupported_params:
            if param in trainer_cfg_dict:
                del trainer_cfg_dict[param]
        
        # Create OmegaConf from dict (this preserves nested structure)
        trainer_cfg = OmegaConf.create(trainer_cfg_dict)
        
        # The trainer needs data, model, logging, checkpoint, etc.
        # data is at top level (cfg.data), others are in cfg.trainer
        trainer_cfg["data"] = cfg.data
        
        # The trainer will handle recursive instantiation internally
        # Use _recursive_=False and let the trainer instantiate nested configs
        trainer = instantiate(trainer_cfg, _recursive_=False)
        trainer.run()
    else:
        # Fall back to simple training loop
        loss_fn = instantiate(cfg.trainer.loss)["all"]
        optimizer = None
        scheduler = None
        use_amp = True
        amp_dtype = None
        if "trainer" in cfg and "optim" in cfg.trainer:
            optim_conf = cfg.trainer.optim
            if "options" in optim_conf:
                _ensure_interval_scaling(optim_conf.options)
            if "amp" in optim_conf:
                if "enabled" in optim_conf.amp:
                    use_amp = bool(optim_conf.amp.enabled)
                if "amp_dtype" in optim_conf.amp:
                    amp_dtype = get_amp_type(optim_conf.amp.amp_dtype)
            trainable_params = {
                name for name, p in model.named_parameters() if p.requires_grad
            }
            optimizer = construct_optimizer(
                model=model,
                optimizer_conf=optim_conf.optimizer,
                options_conf=optim_conf.get("options", None),
                param_group_modifiers_conf=optim_conf.get("param_group_modifiers", None),
                param_allowlist=trainable_params,
                validate_param_groups=False,
            )
            _validate_optimizer_params(optimizer, model, trainable_params)

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            ckpt_dir=cfg.trainer.checkpoint.save_dir,
            num_epochs=cfg.scratch.num_epochs,
            base_lr=cfg.scratch.base_lr,
            vision_lr=cfg.scratch.get("vision_lr", None),  # vision_lr may not exist in all configs
            accumulation_steps=4,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume_path=resume_path,
            early_stop=True,
            early_stop_patience=10,
        )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        resume_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default config
        config_path = "/home/mengnan/seg3d/sam2/sam2/configs/sam2.1_training/sam2.1_multiview_finetune.yaml"
        resume_path = None
    
    # If resume_path not specified, check for best.pt if using resume config
    if resume_path is None and "resume" in config_path.lower():
        best_ckpt = "/home/mengnan/seg3d/checkpoints/best.pt"
        if os.path.exists(best_ckpt):
            resume_path = best_ckpt
            print(f"📂 Auto-detected resume config: Will resume from {resume_path}")
    
    main(config_path, resume_path=resume_path)
