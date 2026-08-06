import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
csv, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
title = sys.argv[4] if len(sys.argv) > 4 else ""
bb=None; xs=[]; ys=[]; ps=[]
for ln in open(csv):
    if ln.startswith("#bbox"): bb=[float(v) for v in ln.split()[1:5]]; continue
    if ln.startswith("x,") or not ln.strip(): continue
    x,y,p,a = ln.split(","); xs.append(float(x)); ys.append(float(y)); ps.append(float(p))
xs=np.array(xs); ys=np.array(ys); ps=np.array(ps)
x0,y0,x1,y1 = bb if bb else [xs.min(),ys.min(),xs.max(),ys.max()]
grid=np.zeros((N,N))
ix=np.clip(((xs-x0)/(x1-x0)*N).astype(int),0,N-1)
iy=np.clip(((ys-y0)/(y1-y0)*N).astype(int),0,N-1)
np.add.at(grid,(iy,ix),ps)
tile_area=((x1-x0)/N)*((y1-y0)/N)
dens=grid/tile_area                       # power per unit area
fig,ax=plt.subplots(figsize=(7,6))
vmax=np.percentile(dens[dens>0],99) if (dens>0).any() else dens.max()
im=ax.imshow(dens,vmin=0,vmax=vmax,origin='lower',extent=[x0,x1,y0,y1],cmap='jet',interpolation='bilinear',aspect='equal')
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label('power density (pW/um^2, 4x); clipped @ 99th pct')
ax.set_xlabel('x (um, 4x)'); ax.set_ylabel('y (um, 4x)')
ax.set_title(title+f"  ({N}x{N} grid, sum P per tile)")
fig.tight_layout(); fig.savefig(out,dpi=130); print("wrote",out,"grid sum=%.3g max_tile=%.3g"%(grid.sum(),dens.max()))
