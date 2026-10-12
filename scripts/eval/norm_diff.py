"""Compare observation normalizers and network weights between two checkpoints."""
import sys, torch

A, B = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
LAYOUT = [("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3), ("phase", 2),
          ("joint_pos", 23), ("joint_vel", 23), ("actions", 13), ("height_command", 1)]
names = [f"{n}[{i}]" for n, k in LAYOUT for i in range(k)]
for net in ("actor_state_dict", "critic_state_dict"):
  sa, sb = A[net], B[net]
  keys = [k for k in sa if k.startswith("obs_normalizer")]
  print(f"\n{net}: normalizer keys {keys}")
  ca, cb = sa.get("obs_normalizer.count"), sb.get("obs_normalizer.count")
  print(f"  count {int(ca)} -> {int(cb)}")
  sda, sdb = sa["obs_normalizer._std"].flatten(), sb["obs_normalizer._std"].flatten()
  ma, mb = sa["obs_normalizer._mean"].flatten(), sb["obs_normalizer._mean"].flatten()
  ratio = sdb / sda
  shift = (mb - ma).abs() / sda
  print(f"  std ratio  min {ratio.min():.4g} max {ratio.max():.4g}; mean shift (in old std) max {shift.max():.4g}")
  idx = torch.argsort(-(ratio.log().abs() + shift))[:8]
  for i in idx.tolist():
    n = names[i] if net == "actor_state_dict" and i < len(names) else f"dim {i}"
    print(f"    {n:>22}: std {sda[i]:.4g} -> {sdb[i]:.4g}  mean {ma[i]:.4g} -> {mb[i]:.4g}")
  wk = [k for k in sa if not k.startswith("obs_normalizer") and sa[k].dtype.is_floating_point]
  rel = max(float((sb[k] - sa[k]).norm() / (sa[k].norm() + 1e-12)) for k in wk)
  print(f"  max relative weight change over layers: {rel:.3e}")
