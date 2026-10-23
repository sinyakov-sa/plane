"""MG20 Tail RevD — parametric preliminary construction. Units millimeters.
Reproduces solids, laser layout trials, hinge axis datum, motion-clearance report.
Dependencies: cadquery, OCP, shapely, ezdxf, numpy, matplotlib.
THIS IS NOT AN AIRWORTHINESS CERTIFICATION OR CUT-READY FLIGHT DESIGN.
"""
from pathlib import Path
import math, json, csv, collections
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import rotate
import cadquery as cq
from cadquery import Vector
import ezdxf
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepCheck import BRepCheck_Analyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon

R=Path(__file__).resolve().parent
for f in ('step','cut','previews','tables'): (R/f).mkdir(exist_ok=True)
P=dict(hs_span=1000,hs_root_c=200,hs_tip_c=120,hs_tip_le=56,
       vs_zmin=15,vs_zmax=435,vs_root_c=300,vs_tip_c=180,vs_tip_le=84,
       hs_t=.10,vs_t=.09,hinge_fraction=.7,gap=2,
       elevator_centre_gap=30,
       hs_hinge_x=140,vs_hinge_x=210,
       hs_max_deflection_deg=25,vs_max_deflection_deg=30,
       elevator_le_bevel_fraction=.08,
       main_cap_fraction=.30,rear_cap_fraction=.57,
       main_cap_size=[5,4],rear_cap_size=[3,3],
       cap_skin_allowance=.35,web_ply_thickness=2,
       notch_clearance_nominal=.2,
       balsa_rib_thickness=3,ply_rib_thickness=2,
       hs_fixed_stations=[15,120,220,320,455,485],
       hs_control_stations=[35,120,220,320,455,485],
       vs_rib_stations=[18,95,175,255,335,390,420],
       hs_hinge_stations=[120,320,455],
       vs_hinge_stations=[95,255,390],
       fixed_nose_fraction=.08)
(R/'parameters.json').write_text(json.dumps(P,ensure_ascii=False,indent=2),encoding='utf-8')

# STEP coordinate system: +X aft; +Y right wing; +Z up.
def dims(axis,s):
    if axis=='H':
        t=abs(s)/500
        return (200-80*t,56*t,.10)
    t=(s-15)/420
    return (300-120*t,84*t,.09)

def naca(x,c,t):
    u=min(1,max(0,x/c))
    return max(0,5*t*c*(.2969*math.sqrt(u)-.126*u-.3516*u*u+.2843*u**3-.1036*u**4))

def cutoff(axis,kind,c):
    hinge=P['hinge_fraction']*c
    if kind=='fixed': return 0,hinge-P['gap']/2
    return hinge+P['gap']/2,c

def shape2d(axis,s,kind,n=100,part='full',rib=False):
    c,le,t=dims(axis,s)
    lo,hi=cutoff(axis,kind,c)
    if part=='nose':
        assert kind=='fixed'
        hi=.08*c
    elif part=='bevel':
        assert kind=='control'
        hi=min(c,lo+.08*c)
    elif rib:
        lo=max(lo,(.08*c if kind=='fixed' else lo+.08*c))
    xs=lo+(hi-lo)*(1-np.cos(np.linspace(0,math.pi,n)))/2
    heights=[]
    for x in xs:
        y=naca(x,c,t)
        if kind=='control':
            y*=max(.05,min(1,(x-(.70*c+P['gap']/2))/(0.08*c)))
        heights.append(float(y))
    coords=[(float(le+x),y) for x,y in zip(xs,heights)]
    coords += [(float(le+x),-y) for x,y in zip(reversed(xs),reversed(heights))]
    poly=Polygon(coords).buffer(0)
    assert poly.is_valid and poly.geom_type=='Polygon' and poly.area>0, (axis,s,kind,part,rib)
    return poly

def w3(pts,axis,s):
    if axis=='H': p=[Vector(x,s,z) for x,z in pts]
    else: p=[Vector(x,z,s) for x,z in pts]
    return cq.Wire.makePolygon(p+[p[0]])

def loft(poly_fn,axis,stations):
    shape=cq.Solid.makeLoft([w3(list(poly_fn(st).exterior.coords)[:-1],axis,st) for st in stations],ruled=True)
    assert shape.isValid() and shape.Volume()>0
    return shape

def export(shape,p): cq.exporters.export(shape,str(p),exportType='STEP')

def cap_xy(axis,s,frac,w,height,up):
    c,le,t=dims(axis,s)
    x=le+frac*c
    outer=naca(frac*c,c,t)-P['cap_skin_allowance']
    inside=outer-height
    assert inside>.1,(axis,s,frac,inside)
    return [(x-w/2,inside),(x+w/2,inside),(x+w/2,outer),(x-w/2,outer)] if up else [(x-w/2,-outer),(x+w/2,-outer),(x+w/2,-inside),(x-w/2,-inside)]

# Actual aerodynamic envelopes; separated from construction; not physical full-block parts.
S={}
for side,mul in [('L',-1),('R',1)]:
    S[f'HS_{side}_Fixed']=loft(lambda st:shape2d('H',st,'fixed'),'H',[0,mul*500])
    S[f'HS_{side}_Elevator']=loft(lambda st:shape2d('H',st,'control'),'H',[mul*15,mul*500])
S['VS_Fin']=loft(lambda st:shape2d('V',st,'fixed'),'V',[15,435])
S['VS_Rudder']=loft(lambda st:shape2d('V',st,'control'),'V',[15,435])
for n,s in S.items():export(s,R/'step'/f'{n}.step')
export(cq.Compound.makeCompound(list(S.values())),R/'step'/'MG20_RevD_Envelope_6Solids.step')

# Structural members (caps, segmented shear webs, leading strips, ribs).
D={}
ribs=[]
web_flat=[]
member_table=[]
for axis,label,start,end in [('H','HS_L',0,-500),('H','HS_R',0,500),('V','VS',15,435)]:
    for name,frac,w,h in [('MAIN',.30,5,4),('REAR',.57,3,3)]:
        for up in [True,False]:
            obj=cq.Solid.makeLoft([w3(cap_xy(axis,s,frac,w,h,up),axis,s) for s in [start,end]],ruled=True)
            assert obj.isValid()
            key=f'{label}_{name}_{"UP" if up else "DOWN"}_CAP'
            D[key]=obj
            member_table.append([key,'cap',f'{w}x{h} mm nominal',abs(end-start),'nominal shaped cap; strength unverified'])
    # Nose (front .08 chord) in upper and lower continuous loft; custom-sanded balsa blank, not laser cut.
    D[f'{label}_BalsaNose']=loft(lambda st:shape2d(axis,st,'fixed',part='nose'),axis,[start,end])
    member_table.append([f'{label}_BalsaNose','leading edge','profiled solid balsa',abs(end-start),'custom shaping; not laser-cut'])

for axis,label,start,end in [('H','EL_L',-15,-500),('H','EL_R',15,500),('V','RUD',15,435)]:
    D[f'{label}_BevelNose']=loft(lambda st:shape2d(axis,st,'control',part='bevel'),axis,[start,end])
    member_table.append([f'{label}_BevelNose','control leading edge','profiled balsa bevel',abs(end-start),'bevel nose; motion clearance checked sectionally'])

# Cap reliefs in cut rib; nose regions omitted (occupied by continuous balsa leading strips).
def rib_poly(axis,s,kind):
    c,le,t=dims(axis,s)
    poly=shape2d(axis,s,kind,rib=True,n=140)
    if kind=='fixed':
        for frac,w,h in [(.30,5,4),(.57,3,3)]:
            xc=le+frac*c
            outer=naca(frac*c,c,t)-P['cap_skin_allowance']
            seat=outer-h
            notch=w+P['notch_clearance_nominal']
            poly=poly.difference(box(xc-notch/2,seat,xc+notch/2,100))
            poly=poly.difference(box(xc-notch/2,-100,xc+notch/2,-seat))
    poly=poly.buffer(0)
    assert poly.geom_type=='Polygon' and poly.is_valid and poly.area>0,(axis,s,kind,poly.geom_type)
    return poly

def add_rib(axis,s,kind,ply,name):
    poly=rib_poly(axis,s,kind)
    thick=2 if ply else (3 if kind=='fixed' else 2)
    cut_axis='H' if axis=='H' else 'V'
    shape=cq.Solid.extrudeLinear(w3(list(poly.exterior.coords)[:-1],cut_axis,s-thick/2),[],Vector(0,thick,0) if cut_axis=='H' else Vector(0,0,thick))
    assert shape.isValid()
    D[f'RIB_{name}']=shape
    ribs.append(dict(name=name,axis=axis,station=s,kind=kind,material='PLY2' if ply else ('BALSA3' if kind=='fixed' else 'BALSA2'),thickness=thick,polygon=poly))

for mul,side in [(-1,'L'),(1,'R')]:
    for s in P['hs_fixed_stations']:
        ply=(s==15 or s in P['hs_hinge_stations'])
        add_rib('H',mul*s,'fixed',ply,f'HS_{side}_F_{s:03d}')
    for s in P['hs_control_stations']:
        ply=s in P['hs_hinge_stations']
        add_rib('H',mul*s,'control',ply,f'HS_{side}_E_{s:03d}')
for s in P['vs_rib_stations']:
    ply=(s==18 or s in P['vs_hinge_stations'])
    add_rib('V',s,'fixed',ply,f'VS_F_{s:03d}')
    add_rib('V',s,'control',s in P['vs_hinge_stations'],f'VS_R_{s:03d}')

# Webs each span interval BUTT against rib faces, avoiding duplicated material in rib.
# Web occupies x half-width 1 mm behind/between the cap rails, vertical span from inner faces of cap top to cap bottom.
for axis,label,station_list in [('H','HS_L',[-s for s in P['hs_fixed_stations']]),('H','HS_R',P['hs_fixed_stations']),('V','VS',P['vs_rib_stations'])]:
    station_list=sorted(station_list)
    for typ,frac,capw,caph in [('MAIN',.30,5,4),('REAR',.57,3,3)]:
        for ia,(sa,sb) in enumerate(zip(station_list,station_list[1:]),start=1):
            ra=2 if axis=='V' and sa in [18,95,255,390] else (2 if axis=='H' and abs(sa) in [15,120,320,455] else 3)
            rb=2 if axis=='V' and sb in [18,95,255,390] else (2 if axis=='H' and abs(sb) in [15,120,320,455] else 3)
            a=sa+ra/2
            b=sb-rb/2
            assert a<b,(label,sa,sb,a,b)
            pts=[]
            for st in (a,b):
                c,le,t=dims(axis,st)
                x=le+frac*c
                surface=naca(frac*c,c,t)-P['cap_skin_allowance']
                inside=surface-caph
                assert inside>0.2,(label,typ,st,inside)
                pts.append((st,x,inside))
            wires=[]
            for st,x,inside in pts:
                coordinates=[(x-1,-inside),(x+1,-inside),(x+1,inside),(x-1,inside)]
                wires.append(w3(coordinates,axis,st))
            solid=cq.Solid.makeLoft(wires,ruled=True)
            assert solid.isValid()
            key=f'{label}_{typ}_WEB_{ia:02d}'
            D[key]=solid
            # Contour of shear web seen from its broad face (station vs section-thickness coordinate).
            poly=Polygon([(a,-pts[0][2]),(b,-pts[1][2]),(b,pts[1][2]),(a,pts[0][2])])
            assert poly.is_valid and poly.area>0
            web_flat.append(dict(name=key,polygon=poly,material='PLY2',span_mm=round(b-a,2),height1_mm=round(2*pts[0][2],2),height2_mm=round(2*pts[1][2],2)))
            member_table.append([key,'shear web','birch plywood 2 mm',round(b-a,2),'trial cut; grain along span preferred; butt joint to ribs'])

# Solid markers for hinge axes: tiny local pin indicators are NOT manufactured hinges.
markers={}
hinges=[]
for mul,side in [(-1,'L'),(1,'R')]:
    for st in P['hs_hinge_stations']:
        y=mul*st
        x=P['hs_hinge_x']
        marker=cq.Solid.makeCylinder(.8,12,Vector(x,y-6,0),Vector(0,1,0))
        key=f'HS_{side}_HINGE_AXIS_{st:03d}'
        markers[key]=marker
        hinges.append([key,x,y,0,'axis Y','pin marker only; hardware not specified'])
for z in P['vs_hinge_stations']:
    key=f'VS_HINGE_AXIS_{z:03d}'
    markers[key]=cq.Solid.makeCylinder(.8,12,Vector(P['vs_hinge_x'],0,z-6),Vector(0,0,1))
    hinges.append([key,P['vs_hinge_x'],0,z,'axis Z','pin marker only; hardware not specified'])

export(cq.Compound.makeCompound(list(D.values())),R/'step'/'MG20_RevD_Structure.step')
export(cq.Compound.makeCompound(list(markers.values())),R/'step'/'MG20_RevD_Hinge_Datums_ONLY.step')

# Section motion-clearance sweep, in continuous idealized smooth-surface geometry (excluding external hardware).
clearance=[]
for axis,samples,amax in [('H',np.linspace(0,500,41),P['hs_max_deflection_deg']),('V',np.linspace(15,435,41),P['vs_max_deflection_deg'])]:
    for s in samples:
        f=shape2d(axis,float(s),'fixed',n=140)
        ctrl=shape2d(axis,float(s),'control',n=140)
        c,le,_=dims(axis,s)
        pivot=P['hs_hinge_x'] if axis=='H' else P['vs_hinge_x']
        assert abs(le+P['hinge_fraction']*c-pivot)<.00001,(axis,s,pivot)
        for angle in np.linspace(-amax,amax,25):
            rotated=rotate(ctrl,float(angle),origin=(pivot,0))
            int_area=f.intersection(rotated).area
            dist=f.distance(rotated)
            clearance.append(dict(axis=axis,station_mm=round(float(s),3),deflection_deg=round(float(angle),2),overlap_mm2=round(float(int_area),7),clearance_mm=round(float(dist),5)))
assert max(x['overlap_mm2'] for x in clearance)<.000001
mincl=min(x['clearance_mm'] for x in clearance)

# DXF cut sheets (each polygon individual closed LWPOLYLINE). True mm; no kerf compensation.
def sheet(parts,name):
    doc=ezdxf.new('R2010')
    doc.header['$INSUNITS']=4
    doc.layers.new('CUT',dxfattribs={'color':1})
    doc.layers.new('PART_ID_DO_NOT_CUT',dxfattribs={'color':5})
    space=doc.modelspace(); cursor=10
    for label,poly in parts:
        xmin,ymin,xmax,ymax=poly.bounds
        points=[(float(x-xmin+10),float(y-ymin+cursor)) for x,y in list(poly.exterior.coords)[:-1]]
        space.add_lwpolyline(points,close=True,dxfattribs={'layer':'CUT'})
        space.add_text(label,dxfattribs={'layer':'PART_ID_DO_NOT_CUT','height':2.5,'insert':(10,cursor+ymax-ymin+2.5)})
        cursor+=ymax-ymin+17
    path=R/'cut'/name
    doc.saveas(path)
    return len(parts)
for material in ['PLY2','BALSA3','BALSA2']:
    count=sheet([(e['name'],e['polygon']) for e in ribs if e['material']==material],f'RevD_Ribs_{material}_TRIAL_1to1.dxf')
    print(material,'rib count',count)
sheet([(e['name'],e['polygon']) for e in web_flat],'RevD_Shear_Webs_PLY2_TRIAL_1to1.dxf')

with (R/'tables'/'Hinges_Datums.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['part','x_mm','y_mm','z_mm','nominal_axis','note']);w.writerows(hinges)
with (R/'tables'/'Rib_Cutlist.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['name','material','thickness_mm','position_axis','station_mm','kind'])
    for e in ribs: w.writerow([e['name'],e['material'],e['thickness'],e['axis'],e['station'],e['kind']])
with (R/'tables'/'Shear_Web_Cutlist.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['name','material','span_length_mm','root_height_mm','tip_height_mm'])
    for e in web_flat:w.writerow([e['name'],e['material'],e['span_mm'],e['height1_mm'],e['height2_mm']])
with (R/'tables'/'Other_Members.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['name','type','stock','length_mm','note']);w.writerows(member_table)
with (R/'tables'/'Clearance_Sweep.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['axis','station_mm','deflection_deg','overlap_mm2','clearance_mm']);w.writeheader();w.writerows(clearance)

# Simple geometry drawing (plan side views), axes intentionally note that this is not a physical assembly on fuselage.
fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
ax=axs[0]
for mul in [-1,1]:
    for kind,color in [('fixed','#2375af'),('control','#d67f20')]:
        poly_st=[]
        for st in [0,500] if kind=='fixed' else [15,500]:
            s=st*mul
            c,le,t=dims('H',s)
            l,h=cutoff('H',kind,c)
            poly_st.append((s,le+l,le+h))
        ax.fill([poly_st[0][0],poly_st[1][0],poly_st[1][0],poly_st[0][0]],
                [poly_st[0][1],poly_st[1][1],poly_st[1][2],poly_st[0][2]],color=color,alpha=.16)
        ax.plot([p[0] for p in poly_st],[p[1] for p in poly_st],color=color)
        ax.plot([p[0] for p in poly_st],[p[2] for p in poly_st],color=color)
    for st in P['hs_hinge_stations']:
        ax.plot([mul*st]*2,[P['hs_hinge_x']-6,P['hs_hinge_x']+6],color='red',lw=2.5)
ax.axhline(140,color='red',ls='--',lw=1,label='elevator axis X=140')
ax.set(xlabel='Y / mm',ylabel='X aft / mm',title='RevD horizontal tail, plan | six hinges on common straight axis')
ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim');ax.legend()
ax=axs[1]
for kind,color in [('fixed','#2375af'),('control','#d67f20')]:
    vals=[]
    for s in [15,435]:
        c,le,t=dims('V',s);l,h=cutoff('V',kind,c)
        vals.append((s,le+l,le+h))
    ax.fill([vals[0][1],vals[1][1],vals[1][2],vals[0][2]],
            [vals[0][0],vals[1][0],vals[1][0],vals[0][0]],color=color,alpha=.16)
    ax.plot([p[1] for p in vals],[p[0] for p in vals],color=color)
    ax.plot([p[2] for p in vals],[p[0] for p in vals],color=color)
for s in P['vs_hinge_stations']:ax.plot([204,216],[s,s],color='red',lw=2.5)
ax.axvline(210,color='red',ls='--',lw=1,label='rudder axis X=210')
ax.set(xlabel='X aft / mm',ylabel='Z up / mm',title='RevD vertical tail | three hinges on vertical axis')
ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim');ax.legend()
fig.savefig(R/'previews'/'MG20_RevD_Tail_Plan.png',dpi=160)
plt.close(fig)

# Motion-clearance plot at root; dashed initial outlines; 25/30deg deflections separately.
fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for ax,axis,st,am in zip(axs,['H','V'],[0,15],[25,30]):
    f=shape2d(axis,st,'fixed')
    ctrl=shape2d(axis,st,'control')
    pivot=140 if axis=='H' else 210
    ax.add_patch(PatchPolygon(list(f.exterior.coords),fc='#90b8d4',ec='#2375af',alpha=.6))
    ax.add_patch(PatchPolygon(list(ctrl.exterior.coords),fc='#f8d4a4',ec='#d67f20',alpha=.5))
    for angle,color in [(-am,'#c65e67'),(am,'#2c9157')]:
        p=rotate(ctrl,angle,origin=(pivot,0))
        a,b=p.exterior.xy
        ax.plot(a,b,color=color,lw=1.4,label=f'{angle:+}°')
    ax.axvline(pivot,color='gray',lw=.8,ls=':')
    ax.set_aspect('equal');ax.set(xlabel='local X, mm',ylabel='profile thickness, mm',title=f'{axis} section: root, ±{am}°')
    ax.grid(alpha=.2); ax.legend()
fig.savefig(R/'previews'/'MG20_RevD_Control_Clearance.png',dpi=180)
plt.close(fig)

# Separate STEP schemas for diagnosis of KOMPAS import quirks.
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
for schema in ('AP203', 'AP214IS'):
    assert Interface_Static.SetCVal_s('write.step.schema', schema)
    writer=STEPControl_Writer()
    writer.Transfer(S['HS_R_Fixed'].wrapped, STEPControl_AsIs)
    writer.Write(str(R/'step'/f'HS_R_Fixed_{schema}_DIAGNOSTIC.step'))

# Occt read-back validates solids & shape not empty. Does NOT validate KOMPAS import.
checks=[]
for path in sorted((R/'step').rglob('*.step')):
    reader=STEPControl_Reader(); stat=reader.ReadFile(str(path)); nroots=reader.TransferRoots(); sh=reader.OneShape()
    obj=cq.Shape(sh); parts=obj.Solids(); vol=sum(s.Volume() for s in parts)
    valid=BRepCheck_Analyzer(sh).IsValid()
    checks.append({'path':str(path.relative_to(R)),'n_solids':len(parts),'volume_mm3':round(vol,2),'occt_valid':bool(valid),'file_bytes':path.stat().st_size})
    assert len(parts)>0 and vol>0 and valid,(path,len(parts),vol,valid,stat,nroots)
assert next(x for x in checks if x['path'].endswith('Envelope_6Solids.step'))['n_solids']==6
assert next(x for x in checks if x['path'].endswith('Structure.step'))['n_solids']==len(D)
assert next(x for x in checks if x['path'].endswith('Datums_ONLY.step'))['n_solids']==len(markers)
summary=dict(revision='RevD',ribs=len(ribs),webs=len(web_flat),caps=12,nose_segments=6,
             modeled_structural_parts=len(D),hinge_datums=len(markers),
             step_files=len(checks),motion_section_samples=len(clearance),
             min_section_clearance_mm=round(mincl,3),max_section_overlap_mm2=max(x['overlap_mm2'] for x in clearance),
             scope='cross-section surface check only; not hardware, not aeroelastic or full 3D clearance',
             readback='OCCT readback; KOMPAS not tested',step_results=checks)
(R/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('RESULT', json.dumps({k:v for k,v in summary.items() if k!='step_results'},ensure_ascii=False))
