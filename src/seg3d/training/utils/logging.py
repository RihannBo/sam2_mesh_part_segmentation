import csv
import os
import torch
import torch.nn as nn


class CSVLogger:
    def __init__(self, log_dir: str, filename: str = "log.csv", resume_from_epoch: int = None):
        """
        CSV Logger for training metrics.
        
        Args:
            log_dir: Directory to save log file
            filename: Log filename (default: "log.csv")
            resume_from_epoch: If provided, truncate log at this epoch before resuming
                              (removes entries >= resume_from_epoch to avoid duplicates)
        """
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        is_new = not os.path.exists(self.path)
        
        # If resuming, truncate log at resume_from_epoch
        if resume_from_epoch is not None and not is_new:
            self._truncate_at_epoch(resume_from_epoch)
            is_new = False  # File exists but was truncated
        
        self.file = open(self.path, "a", newline="")
        self.writer = csv.writer(self.file)

        if is_new:
            self.writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_iou",
                "best_val_loss",
                "lr",
                "epoch_time_sec",
            ])
            self.file.flush()
    
    def _truncate_at_epoch(self, epoch: int):
        """Truncate log file to keep only entries with epoch < resume_from_epoch."""
        import csv as csv_module
        rows_to_keep = []
        with open(self.path, "r", newline="") as f:
            reader = csv_module.reader(f)
            header = next(reader)  # Read header
            rows_to_keep.append(header)
            for row in reader:
                if row and len(row) > 0:
                    try:
                        row_epoch = int(row[0])
                        if row_epoch < epoch:
                            rows_to_keep.append(row)
                    except (ValueError, IndexError):
                        # Keep non-epoch rows (shouldn't happen, but be safe)
                        rows_to_keep.append(row)
        
        # Write back truncated log
        with open(self.path, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerows(rows_to_keep)
        
        print(f"📝 Truncated log.csv: Removed entries from epoch {epoch} onwards (resuming training)")

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_iou: float,
        best_val_loss: float,
        lr: float,
        epoch_time_sec: float,
    ):
        self.writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_iou,
            best_val_loss,
            lr,
            epoch_time_sec,
        ])
        self.file.flush()

    def close(self):
        self.file.close()


def print_parameter_summary(model: nn.Module, verbose: bool = False):
    """
    Print a summary of trainable vs frozen parameters in the model.
    
    Args:
        model: PyTorch model to analyze
        verbose: If True, print detailed breakdown by module
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 80)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 80)
    print(f"Total parameters:      {total_params:>15,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters:   {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen parameters:      {frozen_params:>15,} ({frozen_params/1e6:.2f}M)")
    if total_params > 0:
        trainable_ratio = trainable_params / total_params * 100
        print(f"Trainable ratio:        {trainable_ratio:>15.2f}%")
    print("=" * 80)
    
    if verbose:
        print("\nDetailed breakdown by module:")
        print("-" * 80)
        
        # Group parameters by top-level module
        module_stats = {}
        for name, param in model.named_parameters():
            # Get top-level module name (first component)
            top_module = name.split('.')[0] if '.' in name else name
            if top_module not in module_stats:
                module_stats[top_module] = {'trainable': 0, 'frozen': 0}
            
            if param.requires_grad:
                module_stats[top_module]['trainable'] += param.numel()
            else:
                module_stats[top_module]['frozen'] += param.numel()
        
        for module_name in sorted(module_stats.keys()):
            stats = module_stats[module_name]
            total = stats['trainable'] + stats['frozen']
            if total > 0:
                ratio = stats['trainable'] / total * 100
                print(f"  {module_name:30s} | "
                      f"Trainable: {stats['trainable']:>10,} ({ratio:>5.1f}%) | "
                      f"Frozen: {stats['frozen']:>10,}")
        
        print("-" * 80)
    
    # Verify optimizer will only see trainable params
    optimizer_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if optimizer_params != trainable_params:
        print(f"\n⚠️  WARNING: Mismatch detected! "
              f"Optimizer will see {optimizer_params:,} params, "
              f"but {trainable_params:,} are marked trainable.")
    else:
        print(f"\n✓ Optimizer will optimize {trainable_params:,} parameters")
    
    print("=" * 80 + "\n")