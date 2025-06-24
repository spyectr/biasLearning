# Dataset imports
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import psutil
# record script start time
START_TIME = time.time()
# Additional datasets
from torchvision.datasets import KMNIST, FashionMNIST, QMNIST, SVHN


# QMNIST
class QMNISTDataset(QMNIST):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root=root, train=train,
                         download=download, transform=transform)

# SVHN wrapper to match train/test API
class SVHNWrapper(SVHN):
    def __init__(self, root, train=True, download=True, transform=None):
        split = 'train' if train else 'test'
        super().__init__(root=root, split=split,
                         download=download, transform=transform)


# Reproducibility
SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Control dataset download: pre-cache above, then rely on local cache
DO_DOWNLOAD = False

USE_FULL_KERNEL = True  # if True, use explicit kernel solve, otherwise use CG
# Select NTK mode: 'bias' or 'full'
# NTK_MODE = 'full'  # change to 'full' to run the full-NTK solution
NTK_MODE = 'bias'  # change to 'full' to run the full-NTK solution
# PLOT_FIGS = False  # if True, plot figures

# Device detection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")

# === Memory stats ===
try:
    vm = psutil.virtual_memory()
    print(f"System RAM: total={vm.total/1e9:.2f} GB, available={vm.available/1e9:.2f} GB")
except Exception as e:
    print(f"Could not fetch system RAM info: {e}")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory
    alloc_mem = torch.cuda.memory_allocated(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    print(f"CUDA GPU Memory: total={total_mem/1e9:.2f} GB, allocated={alloc_mem/1e9:.2f} GB, max_allocated={max_alloc/1e9:.2f} GB")
elif device.type == 'mps':
    print("MPS device selected; GPU memory stats not available via torch.")
else:
    usage = psutil.Process().memory_info()
    print(f"Process RSS: {usage.rss/1e9:.2f} GB")

# === Configuration ===
# === Configuration ===
N_CLASSES   = 10
N_PER_CLASS = 20   # MNIST samples per class
N_TEST      = 500  # number of test samples per dataset (None to use full test set)
INPUT1      = 10    # reduced dimension
INPUT_DIM   = INPUT1 * INPUT1  # 5x5 reduced MNIST images
N_HIDDEN    = 2**14  # fixed number of hidden units
# N_HIDDEN    = INPUT_DIM * N_PER_CLASS * N_CLASSES # fixed number of hidden units

# List of (name, constructor) pairs for all datasets
DATASETS = [
    ('MNIST',         datasets.MNIST),
    ('QMNIST',        QMNISTDataset),
    ('KMNIST',        KMNIST),
    ('FashionMNIST',  FashionMNIST),
    ('SVHN',          SVHNWrapper),
]
# Flags: 1 to include, 0 to skip. Length must match DATASETS.
LoadDatasets = [1, 1, 1, 1, 0]
print(f"Using {N_HIDDEN} hidden units")
# print the ratio between hidden units and N_CLASSES*N_PER_CLASS
print(f"Ratio of hidden units to N_CLASSES*N_PER_CLASS: {N_HIDDEN / (N_CLASSES * N_PER_CLASS):.2f}")





# === Conjugate Gradient Solver ===
# A simple conjugate gradient solver for linear systems Ax = b
# Uses a function handle Ax to compute matrix-vector products
def cg_solve(Ax, b, tol=1e-6, maxiter=500):
    x = torch.zeros_like(b)
    r = b - Ax(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    for _ in range(maxiter):
        Ap = Ax(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x += alpha*p
        r -= alpha*Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new/rs_old)*p
        rs_old = rs_new
    return x


def bias_only_ntk_update(x_train, y_train, x_test, y_test, model, epsilon=1e-6):
    """
    Returns (train_acc_lin, test_acc_lin, delta_b)
    """
    # Forward on train
    out_init, h_init = model(x_train)
    # Build residual and Jacobian
    P = x_train.size(0)
    C = y_train.max().item() + 1  # or N_CLASSES
    # Targets: +1 for correct, -1 for others
    y_target = -torch.ones(P, C, device=device)
    y_target.scatter_(1, y_train.unsqueeze(1), 1.0)
    res0 = (y_target - out_init).reshape(-1)
    mask = (h_init > 0).to(x_train.dtype)
    Jb1 = mask.unsqueeze(1) * model.W2.unsqueeze(0)
    Jb2 = torch.eye(C, device=device).unsqueeze(0).expand(P, -1, -1)
    J_bias = torch.cat([Jb1, Jb2], dim=2).reshape(P*C, N_HIDDEN+C)
    # Solve for alpha via CG or explicit based on USE_FULL_KERNEL

    if USE_FULL_KERNEL:
        K = J_bias @ J_bias.T
        K += epsilon * torch.eye(K.size(0), device=device)
        alpha = torch.linalg.solve(K.cpu(), res0.cpu()).to(device)
    else:
        # Local matvec using the function's J_bias
        alpha = cg_solve(lambda v: J_bias @ (J_bias.T @ v) + epsilon * v, res0)    # Compute delta_b and linearized train accuracy
    delta_b = J_bias.T @ alpha
    out_lin = (out_init.reshape(-1) + (J_bias @ delta_b)).view(P, C)
    train_acc_lin = (out_lin.argmax(dim=1) == y_train).float().mean().item()
    # Test set
    out_test_init, h_test_init = model(x_test)
    P_test = x_test.size(0)
    mask_test = (h_test_init > 0).to(x_test.dtype)
    Jb1_t = mask_test.unsqueeze(1) * model.W2.unsqueeze(0)
    Jb2_t = torch.eye(C, device=device).unsqueeze(0).expand(P_test, -1, -1)
    Jb_test = torch.cat([Jb1_t, Jb2_t], dim=2).reshape(P_test*C, N_HIDDEN+C)
    out_test_lin = (out_test_init.reshape(-1) + (Jb_test @ delta_b)).view(P_test, C)
    test_acc_lin = (out_test_lin.argmax(dim=1) == y_test).float().mean().item()
    return train_acc_lin, test_acc_lin, delta_b


# === Full-NTK Update ===
def full_ntk_update(x_train, y_train, x_test, y_test, model, epsilon=1e-6):
    """
    Runs full-NTK update on all parameters (W1, b1, W2, b2).
    Returns (train_acc_lin, test_acc_lin, delta_theta).
    """
    # Forward on train
    out_init, h_init = model(x_train)
    P = x_train.size(0)
    C = y_train.max().item() + 1
    # Build +1/-1 targets
    y_target = -torch.ones(P, C, device=device)
    y_target.scatter_(1, y_train.unsqueeze(1), 1.0)
    res0 = (y_target - out_init).reshape(-1)
    # Precompute inputs and mask
    x_flat = x_train.view(P, -1)                            # [P, INPUT_DIM]
    mask = (h_init > 0).to(x_train.dtype)                   # [P, N_HIDDEN]
    # dOut/dW1
    temp = mask.unsqueeze(1) * model.W2.unsqueeze(0)        # [P, C, N_HIDDEN]
    Jw1 = (temp.unsqueeze(-1) * x_flat.unsqueeze(1).unsqueeze(1)) \
          .reshape(P*C, N_HIDDEN*INPUT_DIM)
    # dOut/db1
    Jb1 = temp.reshape(P*C, N_HIDDEN)
    # dOut/dW2
    Jw2 = h_init.unsqueeze(1).expand(P, C, N_HIDDEN) \
          .reshape(P*C, N_HIDDEN)
    # dOut/db2
    Jb2 = torch.eye(C, device=device).unsqueeze(0) \
          .expand(P, -1, -1).reshape(P*C, C)
    # Full Jacobian
    J_full = torch.cat([Jw1, Jb1, Jw2, Jb2], dim=1)        # [P*C, total_params]
    # Solve for alpha
    if USE_FULL_KERNEL:
        K = J_full @ J_full.T
        K += epsilon * torch.eye(K.size(0), device=device)
        alpha = torch.linalg.solve(K.cpu(), res0.cpu()).to(device)
    else:
        alpha = cg_solve(lambda v: J_full @ (J_full.T @ v) + epsilon * v, res0)
    # Parameter update vector
    delta_theta = J_full.T @ alpha
    # Linearized train output
    lin_flat = out_init.reshape(-1) + (J_full @ delta_theta)
    out_lin = lin_flat.view(P, C)
    train_acc_lin = (out_lin.argmax(dim=1) == y_train).float().mean().item()
    # Test set
    out_test_init, h_test_init = model(x_test)
    P_test = x_test.size(0)
    x_test_flat = x_test.view(P_test, -1)
    mask_t = (h_test_init > 0).to(x_test.dtype)
    temp_t = mask_t.unsqueeze(1) * model.W2.unsqueeze(0)
    Jw1_t = (temp_t.unsqueeze(-1) * x_test_flat.unsqueeze(1).unsqueeze(1)) \
             .reshape(P_test*C, N_HIDDEN*INPUT_DIM)
    Jb1_t = temp_t.reshape(P_test*C, N_HIDDEN)
    Jw2_t = h_test_init.unsqueeze(1).expand(P_test, C, N_HIDDEN) \
             .reshape(P_test*C, N_HIDDEN)
    Jb2_t = torch.eye(C, device=device).unsqueeze(0) \
             .expand(P_test, -1, -1).reshape(P_test*C, C)
    J_full_t = torch.cat([Jw1_t, Jb1_t, Jw2_t, Jb2_t], dim=1)
    lin_flat_test = out_test_init.reshape(-1) + (J_full_t @ delta_theta)
    out_test_lin = lin_flat_test.view(P_test, C)
    test_acc_lin = (out_test_lin.argmax(dim=1) == y_test).float().mean().item()
    return train_acc_lin, test_acc_lin, delta_theta


# === Multi‐Dataset Loading & Subsampling ===
transform = transforms.Compose([
    transforms.Resize((INPUT1, INPUT1)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# === Multi‐Dataset Loading & Subsampling (Optimized with DataLoader) ===
train_sets, test_sets, names = [], [], []
for flag, (name, Dclass) in zip(LoadDatasets, DATASETS):
    # Skip if not selected or not available
    if not flag:
        continue
    if flag and Dclass is None:
        print(f"Warning: {name} selected but not available—skipping.")
        continue
    # --- Train split: bulk load then subsample via tensor indexing ---
    full_tr = Dclass(root='.', train=True, download=DO_DOWNLOAD, transform=transform)
    # Load entire train split once
    loader_full_tr = DataLoader(full_tr, batch_size=len(full_tr), shuffle=False)
    (x_all_tr, y_all_tr), = loader_full_tr
    x_all_tr, y_all_tr = x_all_tr.to(device), y_all_tr.to(device)
    # Subsample N_PER_CLASS per class
    idxs = []
    for c in range(N_CLASSES):
        all_idx = torch.where(y_all_tr == c)[0]
        perm = torch.randperm(len(all_idx), generator=torch.Generator().manual_seed(SEED))
        idxs.append(all_idx[perm[:N_PER_CLASS]])
    train_idx = torch.cat(idxs)
    x_tr = x_all_tr[train_idx]
    y_tr = y_all_tr[train_idx]
    train_sets.append((x_tr, y_tr))
    # Clean up
    del full_tr, loader_full_tr, x_all_tr, y_all_tr
    # --- Test split: bulk load then subsample via tensor indexing ---
    full_te = Dclass(root='.', train=False, download=DO_DOWNLOAD, transform=transform)
    loader_full_te = DataLoader(full_te, batch_size=len(full_te), shuffle=False)
    (x_all_te, y_all_te), = loader_full_te
    x_all_te, y_all_te = x_all_te.to(device), y_all_te.to(device)
    # Subsample up to N_TEST
    num_te = x_all_te.size(0)
    if N_TEST is not None and num_te > N_TEST:
        perm_te = torch.randperm(num_te, generator=torch.Generator().manual_seed(SEED))
        idxs_te = perm_te[:N_TEST]
    else:
        idxs_te = torch.arange(num_te, device=device)
    x_te = x_all_te[idxs_te]
    y_te = y_all_te[idxs_te]
    test_sets.append((x_te, y_te))
    # Clean up
    del full_te, loader_full_te, x_all_te, y_all_te
    names.append(name)

# Print loaded datasets
print(f"Loaded datasets: {names}")

# === Model Definition ===
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_HIDDEN, INPUT_DIM) * np.sqrt(1/INPUT_DIM))
        self.b1 = torch.nn.Parameter(torch.zeros(N_HIDDEN))
        self.W2 = torch.nn.Parameter(torch.randn(N_CLASSES, N_HIDDEN) * np.sqrt(1/N_HIDDEN))
        self.b2 = torch.nn.Parameter(torch.zeros(N_CLASSES))
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(F.linear(x, self.W1, self.b1))
        out = F.linear(h, self.W2, self.b2)
        return out, h

# === Per-dataset NTK tasks ===
if NTK_MODE == 'full':
    update_fn = full_ntk_update
    mode_name = 'Full-NTK'
else:
    update_fn = bias_only_ntk_update
    mode_name = 'Bias-only NTK'

for (x_tr, y_tr), (x_te, y_te), name in zip(train_sets, test_sets, names):
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    # instantiate fresh model for this dataset
    model_ds = MLP().to(device).eval()
    # initial accuracies
    with torch.no_grad():
        out_tr_init, _ = model_ds(x_tr)
        acc_tr_init = (out_tr_init.argmax(dim=1) == y_tr).float().mean().item()
        out_te_init, _ = model_ds(x_te)
        acc_te_init = (out_te_init.argmax(dim=1) == y_te).float().mean().item()
    print(f"{name} [Init]: train acc {acc_tr_init*100:.2f}%, test acc {acc_te_init*100:.2f}%")
    # NTK update for this dataset
    train_lin, test_lin, delta = update_fn(x_tr, y_tr, x_te, y_te, model_ds)
    print(f"{name} [{mode_name}]: train acc {train_lin*100:.2f}%, test acc {test_lin*100:.2f}%")

# === Timing ===
ELAPSED = time.time() - START_TIME
print(f"Total script elapsed time: {ELAPSED:.2f} seconds")