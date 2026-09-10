import torch, glob
for p in sorted(glob.glob("custom/experiments/mace_dielectric_LOCO_fold3_mft_energy_reg_weight0.01_lr8e-3_patience100_omat/checkpoints/*_epoch-*.pt")):
    ckpt = torch.load(p, map_location="cpu")
    lrs = [g["lr"] for g in ckpt["optimizer"]["param_groups"]]
    print(p.split("/")[-1], lrs[0])
# for p in sorted(glob.glob("custom/experiments/mace_perovskites_LOCO_fold3_ft_lr8e-3_patience100_omat/checkpoints/*_epoch-*.pt")):
#     ckpt = torch.load(p, map_location="cpu")
#     lrs = [g["lr"] for g in ckpt["optimizer"]["param_groups"]]
#     print(p.split("/")[-1], lrs[0])