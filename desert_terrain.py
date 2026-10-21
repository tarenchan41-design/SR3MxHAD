"""Offline procedural dune meshes: one base height field, four relief amplitudes.

Produces Blender-importable OBJ previews, not UE maps or captured trajectories.
"""
import argparse
from pathlib import Path
import numpy as np
from experiment import write_json


def heightfields(size=257, extent_m=1000., seed=20261029, horizontal_scale=6.):
    if size < 2 or extent_m <= 0 or horizontal_scale<=0: raise ValueError("Positive extent/scale and at least two vertices per side required")
    axis=np.linspace(-extent_m/2,extent_m/2,size); x,y=np.meshgrid(axis,axis)
    x=x/horizontal_scale;y=y/horizontal_scale
    phases=np.random.default_rng(seed).uniform(-np.pi,np.pi,3)
    base=(np.sin(2*np.pi*(x/180+.18*np.sin(y/230))+phases[0])
          +.35*np.sin(2*np.pi*(.6*x+y)/310+phases[1])+.15*np.sin(2*np.pi*(x-.8*y)/95+phases[2]))
    base=(base-base.mean())/base.std()
    return axis, np.asarray([0.,10.,25.,50.])[:,None,None]*base


def mesh_slopes(z, spacing):
    a,b,c,d=z[:-1,:-1],z[:-1,1:],z[1:,:-1],z[1:,1:]
    first=np.hypot((b-a)/spacing,(d-b)/spacing)
    second=np.hypot((d-c)/spacing,(c-a)/spacing)
    return np.degrees(np.arctan(np.concatenate((first.ravel(),second.ravel()))))


def generate(args):
    out=Path(args.out)
    if out.exists(): raise ValueError("Use a new terrain preview directory")
    out.mkdir(parents=True); axis,fields=heightfields(args.size,args.extent,args.seed,args.horizontal_scale)
    spacing=axis[1]-axis[0]; metadata=[]
    for i,z in enumerate(fields):
        slope=mesh_slopes(z,spacing)
        name=f"SR3M_Desert_R{i}"; obj=out/f"{name}.obj"
        with obj.open("w",encoding="ascii") as f:
            f.write("# meters; Z up; shared topology across relief levels\n")
            for row,y in enumerate(axis):
                for col,x in enumerate(axis): f.write(f"v {x:.6f} {y:.6f} {z[row,col]:.6f}\n")
            for row in range(len(axis)-1):
                for col in range(len(axis)-1):
                    a=row*len(axis)+col+1;b=a+1;c=a+len(axis);d=c+1
                    f.write(f"f {a} {b} {d}\nf {a} {d} {c}\n")
        metadata.append({"name":name,"height_std_m":float(z.std()),"max_slope_deg":float(slope.max()),
                         "p95_slope_deg":float(np.quantile(slope,.95)),"height_range_m":[float(z.min()),float(z.max())]})
    np.savez(out/"heightfields.npz",axis_m=axis,height_m=fields)
    write_json(out/"terrain.json",{"seed":args.seed,"extent_m":args.extent,"grid_size":args.size,"horizontal_scale":args.horizontal_scale,"units":"m",
        "method":"shared deterministic sinusoidal dune field; amplitudes std 0/10/25/50m",
        "status":"offline_geometry_preview_not_UE_maps", "maps":metadata})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout="constrained")
    for ax,z,m in zip(axes,fields,metadata):
        im=ax.imshow(z,origin="lower",extent=[axis[0],axis[-1],axis[0],axis[-1]],vmin=fields.min(),vmax=fields.max(),cmap="terrain")
        ax.set_title(f"{m['name']}\nstd {m['height_std_m']:.0f}m / slope max {m['max_slope_deg']:.1f}deg")
        ax.set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)");fig.colorbar(im,ax=axes,label="Ground elevation (m)")
    fig.suptitle("Procedural geometry previews: shared layout, increasing relief; not UE renders")
    fig.savefig(out/"relief.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--size",type=int,default=257)
    p.add_argument("--extent",type=float,default=1000.);p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--horizontal-scale",type=float,default=6.)
    generate(p.parse_args())
