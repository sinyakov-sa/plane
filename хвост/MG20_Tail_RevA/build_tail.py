import cadquery as cq, numpy as np, json, math
from pathlib import Path
from export_kompas import export_kompas
P=Path(__file__).resolve().parent
# Coordinates match wing source: X span, Y aft, Z up, mm.
B=3550.; cr=310.; ct=169.
S=B*(cr+ct)/2; MAC=2/3*(cr*cr+cr*ct+ct*ct)/(cr+ct)
def qbar(span,c0,c1,sweep):
 t=np.linspace(0,1,10001); c=c0+(c1-c0)*t
 return float(np.trapezoid((sweep*t+.25*c)*c,t)/np.trapezoid(c,t))
HLE=77.5+800-qbar(500,200,120,56)
VLE=77.5+900-qbar(420,300,180,120)
def box(x0,x1,y0,y1,z0,z1):
 return cq.Solid.makeBox(x1-x0,y1-y0,z1-z0,cq.Vector(x0,y0,z0))
def section(s,c,le,vert=False):
 u=(1-np.cos(np.linspace(0,np.pi,65)))/2
 # NACA 0010, finite trailing edge (0.21% chord total thickness).
 h=5*.10*(.2969*np.sqrt(u)-.1260*u-.3516*u*u+.2843*u**3-.1015*u**4)*c
 def v(a,b): return cq.Vector(b,le+a,s) if vert else cq.Vector(s,le+a,b)
 top=[v(a*c,b) for a,b in zip(u,h)]; bot=[v(a*c,-b) for a,b in zip(u[::-1],h[::-1])]
 return cq.Wire.assembleEdges([cq.Edge.makeSpline(top),cq.Edge.makeLine(top[-1],bot[0]),cq.Edge.makeSpline(bot)])
def loft(a,b):return cq.Solid.makeLoft([a,b],True)
H1=loft(section(0,200,HLE),section(500,120,HLE+56)); H2=H1.mirror('YZ')
H=H1.fuse(H2)
V=loft(section(25,300,VLE,True),section(445,180,VLE+120,True))
# Constant horizontal hinge Y, 2 mm geometric gap; neutral envelopes only.
fixedH=H.intersect(box(-501,501,HLE-1,HLE+139,-50,50))
movH=H.intersect(box(-501,501,HLE+141,HLE+250,-50,50))
# Elevator split and center clearance for fuselage / rudder.
EL=movH.intersect(box(-501,-35,0,1500,-50,50)); ER=movH.intersect(box(35,501,0,1500,-50,50))
# Vertical hinge line from 65% root chord to 65% tip chord.
def vmask(a,b):
 pts=[cq.Vector(-60,VLE+a,24),cq.Vector(-60,VLE+b,446),cq.Vector(-60,1500,446),cq.Vector(-60,1500,24)]
 w=cq.Wire.makePolygon(pts+[pts[0]])
 return cq.Solid.extrudeLinear(w,[],cq.Vector(120,0,0))
fixedV=V.cut(vmask(194,236)); rudder=V.intersect(vmask(196,238))
envelopes={'H_stabilizer':fixedH,'Elevator_L':EL,'Elevator_R':ER,'V_fin':fixedV,'Rudder':rudder}
A=cq.Assembly(name='MG20_Tail_RevA_ENVELOPES')
for i,(n,s) in enumerate(envelopes.items()):A.add(s,name=n,color=cq.Color(*[(.83,.73,.48),(.2,.55,.8),(.2,.55,.8),(.83,.73,.48),(.2,.55,.8)][i]))
export_kompas(A, P/'MG20_Tail_RevA_Envelope.step')
# Concept structure: independent full-depth ribs and spars, ribs trimmed to spars.
parts=[]
def add(name,shape,material):
 if shape.Volume()>0.001:
  assert shape.isValid(),name
  parts.append((name,shape,material))
for name,env in envelopes.items():
 vertical=name in ['V_fin','Rudder']; fixed=name in ['H_stabilizer','V_fin']
 if not vertical:
  ranges=[(HLE+1,HLE+7),(HLE+58,HLE+61),(HLE+135,HLE+139)] if fixed else [(HLE+141,HLE+145),(HLE+193,HLE+200)]
  # TE follows tapered planform using a thin aft shell strip.
  spars=[]
  for j,(a,b) in enumerate(ranges[:(3 if fixed else 1)]):
   s=env.intersect(box(-501,501,a,b,-40,40));add(f'{name}_member_{j}',s,'balsa' if j==0 else 'birch_plywood');spars.append(s)
  if not fixed:
   # Trailing strip at local chord fraction > 96%; constant TE sweep.
   for sign in [-1,1]:
    pts=[cq.Vector(sign*0,HLE+192,-40),cq.Vector(sign*500,HLE+56+115.2,-40),cq.Vector(sign*500,HLE+200,-40),cq.Vector(sign*0,HLE+220,-40)]
    s=env.intersect(cq.Solid.extrudeLinear(cq.Wire.makePolygon(pts+[pts[0]]),[],cq.Vector(0,0,80)))
    if s.Volume()>.001:add(f'{name}_TE_{sign}',s,'balsa');spars.append(s)
  stations=sorted(set([-498,-420,-340,-260,-180,-100,-36,0,36,100,180,260,340,420,498]))
  for x in stations:
   rib=env.intersect(box(x-1,x+1,0,1500,-40,40))
   if rib.Volume()<.001:continue
   for s in spars:
    if s.Volume()>.001:rib=rib.cut(s)
   add(f'{name}_rib_{x}',rib,'birch_plywood' if abs(x)<=36 else 'balsa')
 else:
  # Spanwise spars follow leading edge / hinge lines.
  spars=[]
  for j,(f0,f1) in enumerate([(0.01,.04),(.30,.31),(.63,.646)] if fixed else [(.654,.68),(.96,1.0)]):
   pts=[cq.Vector(-40,VLE+300*f0,25),cq.Vector(-40,VLE+120+180*f0,445),cq.Vector(-40,VLE+120+180*f1,445),cq.Vector(-40,VLE+300*f1,25)]
   s=env.intersect(cq.Solid.extrudeLinear(cq.Wire.makePolygon(pts+[pts[0]]),[],cq.Vector(80,0,0)))
   add(f'{name}_member_{j}',s,'balsa' if j==0 else 'birch_plywood');spars.append(s)
  for z in [27,85,145,205,265,325,385,443]:
   rib=env.intersect(box(-40,40,0,1500,z-1,z+1))
   if rib.Volume()<.001:continue
   for s in spars:
    if s.Volume()>.001:rib=rib.cut(s)
   add(f'{name}_rib_{z}',rib,'birch_plywood' if z==27 else 'balsa')
F=cq.Assembly(name='MG20_Tail_RevA_FRAME_CONCEPT')
for n,s,m in parts:F.add(s,name=n,color=cq.Color(*((.85,.73,.48) if m=='balsa' else (.55,.36,.19))))
export_kompas(F, P/'MG20_Tail_RevA_Frame.step')
# STEP round-trip checks.
checks={}
for file in ['MG20_Tail_RevA_Envelope.step','MG20_Tail_RevA_Frame.step','MG20_Tail_RevA_Envelope_FLAT.step','MG20_Tail_RevA_Frame_FLAT.step']:
 shapes=cq.importers.importStep(str(P/file)).solids().vals();checks[file]={'solids':len(shapes),'all_valid':all(s.isValid() for s in shapes),'volume_mm3':sum(s.Volume() for s in shapes)}
 assert checks[file]['all_valid'] and shapes
params={'units':'mm','wing_area_nominal_mm2':S,'wing_MAC_mm':MAC,'wing_AC_y':77.5,'H_span':1000,'H_chord_root':200,'H_chord_tip':120,'H_LE_sweep':56,'H_area_gross_mm2':160000,'H_arm':800,'H_volume':160000*800/(S*MAC),'H_LE_y':HLE,'V_height':420,'V_root_z':25,'V_chord_root':300,'V_chord_tip':180,'V_LE_sweep':120,'V_area_gross_mm2':100800,'V_arm':900,'V_volume':100800*900/(S*B),'V_LE_y':VLE,'profile':'NACA0010 finite TE','frame_parts':len(parts),'checks':checks}
(P/'parameters_and_validation.json').write_text(json.dumps(params,indent=2))
# Preview actual CAD triangulation, both envelope and frame.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(14,7),facecolor='#f7f5ef')
for k,items in enumerate([[(n,s,'balsa') for n,s in envelopes.items()],parts]):
 ax=fig.add_subplot(1,2,k+1,projection='3d')
 for n,s,m in items:
  vs,ts=s.tessellate(.6,.15);v=np.array([[p.x,p.y,p.z] for p in vs]);tris=v[np.array(ts)]
  col='#c9a866' if m=='balsa' else '#785633'
  if k==0 and ('Elevator' in n or n=='Rudder'):col='#4196b5'
  ax.add_collection3d(Poly3DCollection(tris,facecolor=col,edgecolor='none',alpha=1))
 ax.set_xlim(-510,510);ax.set_ylim(770,1200);ax.set_zlim(-30,460);ax.set_box_aspect((1020,430,490));ax.view_init(24,-62)
 ax.set_xlabel('X / mm');ax.set_ylabel('Y aft / mm');ax.set_zlabel('Z / mm')
 ax.set_title('External geometry' if k==0 else 'Structural layout concept');ax.set_facecolor('#f7f5ef')
fig.suptitle('MG20 | Tail RevA | PRELIMINARY DESIGN',fontsize=19)
fig.tight_layout();fig.savefig(P/'MG20_Tail_RevA_Preview.png',dpi=160)
print(json.dumps(params,indent=2))
