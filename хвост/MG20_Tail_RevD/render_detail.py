"""Detail drawing (does not regenerate CAD)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
R=Path(__file__).resolve().parent
P=json.loads((R/'parameters.json').read_text())
fig,ax=plt.subplots(figsize=(15,5))
for sign in (-1,1):
   vals=[]
   for y in [0,500]:
      le=56*y/500;c=200-80*y/500;vals.append((sign*y,le,c))
   for kind,color in [('fixed','#246997'),('control','#d38033')]:
      yy=[v[0] for v in vals]
      x1=[v[1]+ (0 if kind=='fixed' else .7*v[2]+1) for v in vals]
      x2=[v[1]+ (.7*v[2]-1 if kind=='fixed' else v[2]) for v in vals]
      if kind=='control':yy[0]=15*sign; x1[0]=141; x2[0]=199
      ax.add_patch(Polygon(list(zip(yy,x1))+list(reversed(list(zip(yy,x2)))),edgecolor=color,facecolor=color,alpha=.19))
      ax.plot(yy,x1,color=color);ax.plot(yy,x2,color=color)
   for station in (120,320,455):
      ax.plot([sign*station],[140],marker='o',markersize=6,color='#a83445')
ax.plot([-500,500],[140,140],color='#a83445',ls='--',lw=1)
ax.annotate('Common hinge axis X=140 mm',xy=(0,140),xytext=(0,105),ha='center',arrowprops={'arrowstyle':'->'},fontsize=11)
ax.set_aspect('equal',adjustable='box');ax.set(xlim=(-520,520),ylim=(-12,225),xlabel='Y, mm',ylabel='X (aft), mm',title='MG20 RevD — horizontal stabilizer and split elevator, plan view')
ax.grid(alpha=.2)
fig.savefig(R/'previews'/'MG20_RevD_Horizontal_Plan.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,8))
for kind,color in [('fixed','#246997'),('control','#d38033')]:
 z=[15,435];le=[0,84];c=[300,180]
 x1=[le[i]+(0 if kind=='fixed' else .7*c[i]+1) for i in range(2)]
 x2=[le[i]+(.7*c[i]-1 if kind=='fixed' else c[i]) for i in range(2)]
 ax.add_patch(Polygon(list(zip(x1,z))+list(reversed(list(zip(x2,z)))),facecolor=color,alpha=.19))
 ax.plot(x1,z,color=color);ax.plot(x2,z,color=color)
for z in (95,255,390):ax.plot([210],[z],marker='o',markersize=6,color='#a83445')
ax.plot([210,210],[15,435],color='#a83445',ls='--',lw=1)
ax.set_aspect('equal',adjustable='box');ax.set(xlim=(-20,325),ylim=(-10,460),xlabel='X (aft), mm',ylabel='Z, mm',title='MG20 RevD — vertical fin and rudder, side view')
ax.grid(alpha=.2)
fig.savefig(R/'previews'/'MG20_RevD_Vertical_Side.png',dpi=180,bbox_inches='tight');plt.close(fig)
