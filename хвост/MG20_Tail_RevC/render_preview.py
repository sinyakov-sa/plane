from pathlib import Path
import cadquery as cq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
root=Path(__file__).resolve().parent
fig=plt.figure(figsize=(12,8.5),facecolor='white')
ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
for file in sorted((root/'step/skins').glob('*.step')):
    shape=cq.importers.importStep(str(file)).val()
    vertices,triangles=shape.tessellate(.9)
    vertices=np.array([(p.x,p.y,p.z) for p in vertices])
    faces=np.array(triangles)
    faces_verts=vertices[faces]
    color='#a8c5d1' if 'Fixed' in file.name or 'Fin' in file.name else '#d8aa86'
    mesh=Poly3DCollection(faces_verts,alpha=.86,facecolors=color,edgecolor='none',zsort='average')
    ax.add_collection3d(mesh)
ax.set(xlim=(-5,315), ylim=(-530,530),zlim=(-40,460),xlabel='X, mm (aft)',ylabel='Y, mm',zlabel='Z, mm')
ax.set_box_aspect((320,1060,500),zoom=.83)
ax.view_init(elev=23,azim=-71)
ax.set_title('MG20 — RevC tail geometry (concept; no fuselage)',fontsize=15)
ax.grid(alpha=.2)
fig.savefig(root/'previews/MG20_Tail_RevC_3D.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print('created 3D preview')
