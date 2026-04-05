
import os
import copy
from datetime import datetime
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from atomic_charge_dataset import AtomicChargeDataset
from gnn_model import ChargeGCN
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEFAULT_POSCAR_DIR = "/ocean/projects/cis250151p/shared/Data/PASCAR"
DEFAULT_CHARGE_DIR = "/ocean/projects/cis250151p/shared/Data/CHARGESSS"
DEFAULT_BEST_MODEL_PATH = "best_charge_gcn.pt"
DEFAULT_CHECKPOINT_PATH = "last_charge_gcn.pt"


def get_artifact_path(env_var_name, default_filename):
    configured_path = os.environ.get(env_var_name)
    if configured_path:
        return configured_path

    run_tag = os.environ.get("RUN_TAG")
    if run_tag:
        suffix = run_tag
    else:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            suffix = job_id
        else:
            # Fallback for local/manual runs outside SLURM.
            suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    stem, extension = os.path.splitext(default_filename)
    return f"{stem}_{suffix}{extension}"


def get_env_int(name, default):
    return int(os.environ.get(name, str(default)))


def get_env_float(name, default):
    return float(os.environ.get(name, str(default)))


def get_env_bool(name, default):
    default_value = "1" if default else "0"
    return os.environ.get(name, default_value) == "1"


def evaluate(model, data_loader, device):
    model.eval()
    total_squared_error = 0.0
    total_targets = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predictions = model(batch)
            total_squared_error += F.mse_loss(predictions, batch.y, reduction="sum").item()
            total_targets += batch.y.numel()

    return total_squared_error / max(total_targets, 1)


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_targets = 0

    for batch in tqdm(data_loader, leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += F.mse_loss(predictions.detach(), batch.y, reduction="sum").item()
        total_targets += batch.y.numel()

    return total_loss / max(total_targets, 1)


def load_checkpoint_if_available(model, optimizer, checkpoint_path, device, resume_training):
    best_val_loss = float("inf")
    start_epoch = 1

    if not resume_training or not os.path.exists(checkpoint_path):
        return best_val_loss, start_epoch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError as error:
        print(f"Checkpoint mismatch, starting fresh: {error}", flush=True)
        return best_val_loss, start_epoch

    best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Resuming from epoch {start_epoch}", flush=True)
    return best_val_loss, start_epoch


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    poscar_dir = os.environ.get("POSCAR_DIR", DEFAULT_POSCAR_DIR)
    charge_dir = os.environ.get("CHARGE_DIR", DEFAULT_CHARGE_DIR)
    best_model_path = get_artifact_path("BEST_MODEL_PATH", DEFAULT_BEST_MODEL_PATH)
    checkpoint_path = get_artifact_path("CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
    num_epochs = get_env_int("NUM_EPOCHS", 400)
    batch_size = get_env_int("BATCH_SIZE", 16)
    hidden_channels = get_env_int("HIDDEN_CHANNELS", 192)
    num_layers = get_env_int("NUM_LAYERS", 4)
    dropout = get_env_float("DROPOUT", 0.1)
    learning_rate = get_env_float("LR", 3e-4)
    weight_decay = get_env_float("WEIGHT_DECAY", 1e-4)
    lr_patience = get_env_int("LR_PATIENCE", 12)
    early_stop_patience = get_env_int("EARLY_STOP_PATIENCE", 40)
    resume_training = get_env_bool("RESUME", False)
    require_gpu = get_env_bool("REQUIRE_GPU", False)
    seed = get_env_int("SEED", 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if require_gpu and device.type != "cuda":
        raise RuntimeError(
            "REQUIRE_GPU=1 but CUDA is unavailable. Check node allocation, driver, and torch build."
        )

    set_seed(seed)

    print(f"Using device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"POSCAR_DIR: {poscar_dir}")
    print(f"CHARGE_DIR: {charge_dir}")
    print(f"BEST_MODEL_PATH: {best_model_path}")
    print(f"CHECKPOINT_PATH: {checkpoint_path}")
    print(f"NUM_EPOCHS: {num_epochs}")
    print(f"BATCH_SIZE: {batch_size}")
    print(f"HIDDEN_CHANNELS: {hidden_channels}")
    print(f"NUM_LAYERS: {num_layers}")
    print(f"DROPOUT: {dropout}")
    print(f"LR: {learning_rate}")
    print(f"WEIGHT_DECAY: {weight_decay}")
    print(f"REQUIRE_GPU: {require_gpu}")

    dataset = AtomicChargeDataset(
        root="processed_data",
        poscar_dir=poscar_dir,
        charge_dir=charge_dir,
    )

    train_idx, temp = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=42)

    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)

    model = ChargeGCN(
        in_channels=dataset[0].x.shape[1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-5,
    )
    criterion = torch.nn.MSELoss()

    best_val_loss, start_epoch = load_checkpoint_if_available(
        model,
        optimizer,
        checkpoint_path,
        device,
        resume_training,
    )
    epochs_without_improvement = 0
    best_model_state = copy.deepcopy(model.state_dict()) if start_epoch > 1 else None

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.6f}",
            flush=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss = evaluate(model, test_loader, device)
    print(f"Final Test MSE: {test_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()