"""
Attention mechanisms for plant disease classification.

Contains:
  SEBlock          — channel attention (Squeeze-and-Excitation)
  SpatialAttention — spatial attention (CBAM spatial branch)
  CBAM             — full channel + spatial attention module
  verify_*         — training verification utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (channel attention).

    Learns a per-channel weight vector that rescales the feature maps:
    channels carrying diagnostic signal are amplified, noisy ones suppressed.

    Steps:
      1. Squeeze  — global average pool: (B, C, H, W) → (B, C)
      2. Excite   — two FC layers with bottleneck: C → C//r → C → sigmoid
      3. Scale    — multiply weights back into feature maps (broadcast over H×W)

    Args:
        channels: number of input/output channels (C)
        r:        reduction ratio for the bottleneck (default 16)
                  raise to 32 if you observe overfitting on rare classes
    """
    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        bottleneck      = max(1, channels // r)
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # (B, C, 1, 1) → (B, C)
        s = self.squeeze(x).view(b, c)
        # (B, C) → (B, C, 1, 1)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e


class SpatialAttention(nn.Module):
    """
    Spatial attention branch (CBAM spatial module).

    Learns an H×W importance mask: high values over lesion/disease regions,
    low values over background and healthy tissue.

    Steps:
      1. Compress channels — avg-pool and max-pool across C: 2 × H × W descriptor
         avg captures consensus presence; max captures peak single-channel activation
      2. Learn mask       — 7×7 conv (2 → 1 channel) + sigmoid
         large kernel integrates neighbourhood context when deciding importance
      3. Apply            — multiply mask into all C feature maps (broadcast)

    Args:
        kernel_size: conv kernel for the spatial mask (7 recommended, 3 for very small maps)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both: (B, 1, H, W)
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.max(dim=1, keepdim=True).values
        # (B, 2, H, W) → conv → (B, 1, H, W)
        mask = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Chains channel attention (SEBlock) then spatial attention in sequence.
    Channel-first ordering is empirically better: it selects which feature
    maps matter before spatial attention decides where to look within them.

    Args:
        channels:    number of feature map channels at injection point
        r:           SE reduction ratio (default 16, raise to 32 to reduce overfitting)
        kernel_size: spatial attention conv kernel (default 7)
    """
    def __init__(self, channels: int, r: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = SEBlock(channels, r)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)   # which feature maps matter
        x = self.spatial(x)   # where to look within them
        return x


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def verify_cbam_registered(model: nn.Module) -> bool:
    """
    Level 1 check: confirm CBAM parameters exist and have requires_grad=True.
    Run this immediately after model construction, before any training.

    Returns True if everything looks correct, False if something is wrong.
    """
    cbam_params = {
        name: p for name, p in model.named_parameters()
        if ('channel' in name or 'spatial' in name) and p.requires_grad
    }

    if not cbam_params:
        print("WARNING: No CBAM parameters found with requires_grad=True.")
        print("  Check that block_indices are correct and CBAM was injected.")
        return False

    print(f"CBAM registered — {len(cbam_params)} parameter tensors:\n")
    total = 0
    for name, p in cbam_params.items():
        print(f"  {name:<60s} shape={str(p.shape):<20s} numel={p.numel():,}")
        total += p.numel()
    print(f"\n  Total CBAM parameters: {total:,}")
    return True


def verify_cbam_gradients(model: nn.Module, val_loader, criterion,
                           device: torch.device) -> bool:
    """
    Level 2 check: confirm CBAM parameters receive non-zero gradients
    after a real forward+backward pass.

    Run this after level 1 passes, before full training begins.
    A None gradient means the module is disconnected from the graph.
    A near-zero gradient (< 1e-8) means vanishing — no useful update will happen.

    Returns True if all CBAM parameters have healthy gradients.
    """
    model.train()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    # Temporary optimizer — just to zero grads cleanly
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    opt.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()

    print(f"Gradient check (loss={loss.item():.4f})\n")
    header = f"{'Parameter':<55} {'grad mean':>12} {'grad std':>12} {'status':>10}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for name, p in model.named_parameters():
        if ('channel' in name or 'spatial' in name) and p.requires_grad:
            if p.grad is None:
                print(f"{name:<55} {'None':>12} {'None':>12} {'DEAD':>10}")
                all_ok = False
            else:
                g_mean  = p.grad.abs().mean().item()
                g_std   = p.grad.std().item()
                status  = 'ok' if g_mean > 1e-8 else 'VANISHED'
                if status != 'ok':
                    all_ok = False
                print(f"{name:<55} {g_mean:>12.2e} {g_std:>12.2e} {status:>10}")

    print()
    if all_ok:
        print("All CBAM parameters are receiving healthy gradients.")
    else:
        print("WARNING: Some CBAM parameters have dead/vanished gradients.")
        print("  Check learning rate assignment and nn.Sequential wrapping.")

    model.eval()
    return all_ok


def snapshot_cbam_weights(model: nn.Module) -> dict:
    """
    Level 3 helper: snapshot current CBAM weight values.
    Call before training starts, then call compare_cbam_snapshots after.
    """
    return {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if ('channel' in name or 'spatial' in name) and p.requires_grad
    }


def compare_cbam_snapshots(before: dict, after: dict) -> None:
    """
    Level 3 check: confirm CBAM weights actually changed during training.

    Status codes:
      ok       — healthy update magnitude
      FROZEN   — abs change < 1e-7 — weights didn't move at all
                 (learning rate too low, or param not in optimizer)
      UNSTABLE — relative change > 50% — CBAM learning rate too high,
                 attention modules are thrashing and may destabilise backbone
    """
    header = f"{'Parameter':<55} {'abs change':>12} {'rel change':>12} {'status':>10}"
    print(header)
    print("-" * len(header))

    for name in before:
        if name not in after:
            print(f"{name:<55} {'MISSING in after snapshot':>36}")
            continue

        delta   = (after[name] - before[name]).abs()
        abs_chg = delta.mean().item()
        rel_chg = (delta / (before[name].abs() + 1e-8)).mean().item()

        if abs_chg < 1e-7:
            status = 'FROZEN'
        elif rel_chg > 0.5:
            status = 'UNSTABLE'
        else:
            status = 'ok'

        print(f"{name:<55} {abs_chg:>12.2e} {rel_chg:>12.2%} {status:>10}")


def check_attention_diversity(model: nn.Module, val_loader,
                               device: torch.device, n_batches: int = 5) -> None:
    """
    Level 4 check: confirm attention modules learned non-trivial weights.

    This catches the collapse-to-uniform failure mode where all checks pass
    technically (gradients flow, weights update) but the module converged
    to outputting ~0.5 everywhere — mathematically equivalent to no attention.

    SE verdict criteria:
      std > 0.10 and range > 0.30 → ok (different channels get different weights)
      otherwise → COLLAPSED (uniform weights, module is a no-op)

    Spatial verdict criteria:
      variance > 0.02 → ok (mask focuses on specific regions)
      otherwise → FLAT (uniform mask, module is a no-op)
    """
    model.eval()

    se_stats: dict      = {}
    spatial_stats: dict = {}
    hooks               = []

    for name, module in model.named_modules():
        if isinstance(module, SEBlock):
            def _se_hook(n):
                def hook(_, __, output):
                    se_stats[n] = {
                        'mean': output.mean().item(),
                        'std' : output.std().item(),
                        'min' : output.min().item(),
                        'max' : output.max().item(),
                    }
                return hook
            hooks.append(module.excitation.register_forward_hook(_se_hook(name)))

        if isinstance(module, SpatialAttention):
            def _sp_hook(n):
                def hook(_, __, output):
                    spatial_stats[n] = {
                        'mean': output.mean().item(),
                        'var' : output.var().item(),
                        'min' : output.min().item(),
                        'max' : output.max().item(),
                    }
                return hook
            hooks.append(module.register_forward_hook(_sp_hook(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    # SE report
    print("SE excitation weight distributions")
    print("  Healthy: std > 0.10, range > 0.30")
    hdr = f"  {'Module':<45} {'mean':>6} {'std':>6} {'min':>6} {'max':>6} {'verdict':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in se_stats.items():
        spread  = s['max'] - s['min']
        verdict = 'ok' if s['std'] > 0.10 and spread > 0.30 else 'COLLAPSED'
        print(f"  {name:<45} {s['mean']:>6.3f} {s['std']:>6.3f} "
              f"{s['min']:>6.3f} {s['max']:>6.3f} {verdict:>10}")

    print()

    # Spatial report
    print("Spatial mask variance")
    print("  Healthy: var > 0.02")
    hdr = f"  {'Module':<45} {'mean':>6} {'var':>8} {'min':>6} {'max':>6} {'verdict':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in spatial_stats.items():
        verdict = 'ok' if s['var'] > 0.02 else 'FLAT'
        print(f"  {name:<45} {s['mean']:>6.3f} {s['var']:>8.4f} "
              f"{s['min']:>6.3f} {s['max']:>6.3f} {verdict:>10}")