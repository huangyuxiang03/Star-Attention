import torch

kv_ranks = [None for i in range(8)]
for i in range(8):
    kv_ranks[i] = torch.load(f"{i}.ckpt")
    
kv = []
for l in range(32):
    ks = []
    vs = []
    for i in range(8):
        ks.append(kv_ranks[i][l][0].cuda())
        vs.append(kv_ranks[i][l][1].cuda())
    ks = torch.cat(ks, dim=-2)
    vs = torch.cat(vs, dim=-2)
    kv.append((ks, vs))

std_kv = torch.load("std.ckpt")

for l in range(32):
    print(f"layer {l}")
    k_delta = std_kv[l][0].cuda()[:, :, :16041, :] - kv[l][0]
    v_delta = std_kv[l][1].cuda()[:, :, :16041, :] - kv[l][1]
    k_delta = k_delta.abs().max(dim=0).values.max(dim=0).values.max(dim=-1).values
    v_delta = v_delta.abs().max(dim=0).values.max(dim=0).values.max(dim=-1).values
    
    # breakpoint()
    
    
    print(f"k: {k_delta.max()} / {kv[l][0].abs().max()}\nv: {v_delta.max()} / {kv[l][1].abs().max()}")

breakpoint()
a = 1