"""MG20 tail RevC concept generator; mm. Requires cadquery, scipy, shapely, ezdxf, matplotlib.
Regenerate from directory: python generate.py
"""
from pathlib import Path
import math, json, collections
import numpy as np
import cadquery as cq
from cadquery import Vector
from shapely.geometry import Polygon, box
import ezdxf
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ROOT=Path(__file__).resolve().parent
for folder in ['step/skins','step/structure','cut','previews']:
    (ROOT/folder).mkdir(parents=True, exist_ok=True)

P={
 'hs_span_mm':1000,'hs_croot_mm':200,'hs_ctip_mm':120,'hs_LEtip_mm':40,
 'vs_height_mm':420,'vs_croot_mm':300,'vs_ctip_mm':180,'vs_LEtip_mm':120,
 'vs_base_above_hs_plane_mm':15, 'h_t':0.10,'v_t':0.09,
 'hinge_fraction':0.70,'gap_mm':2.0,'elevator_center_gap_mm':30,
 'hs_main_cap_x_over_c':0.30,'hs_rear_cap_x_over_c':0.57,
 'vs_main_cap_x_over_c':0.30,'vs_rear_cap_x_over_c':0.57,
 'main_cap_width_mm':5, 'main_cap_height_mm':4,
 'rear_cap_width_mm':3, 'rear_cap_height_mm':3,
 'spar_undersize_mm':0.35,
 'laser_notch_extra_mm':0.2,
 'ply_rib_mm':2, 'balsa_fixed_rib_mm':3, 'balsa_control_rib_mm':2,
 'web_material_mm':2,
 'shear_web_segments':'to be designed after joint/station validation',
}
(ROOT/'parameters.json').write_text(json.dumps(P,indent=2,ensure_ascii=False),encoding='utf-8')

def h_station(s):
    s=abs(s)/500
    return (200-80*s,40*s)   # chord, leading edge X

def v_station(z):
    s=(z-15)/420
    return (300-120*s, 120*s)

def naca_half(x,c,t):
    u=max(0.0,min(1.0,x/c))
    return max(0.0, (5*t*c*(0.2969*math.sqrt(u)-0.1260*u-0.3516*u**2+0.2843*u**3-0.1036*u**4)))

def ranges(c,section):
    center=0.70*c
    return (0.0, center-P['gap_mm']/2) if section=='fixed' else (center+P['gap_mm']/2,c)

def profile(c,le,t,kind,n=56):
    a,b=ranges(c,kind)
    # Cosine spacing allows enough samples near narrow leading/trailing ends
    angle=np.linspace(0,math.pi,n)
    xs=a+(b-a)*(1-np.cos(angle))/2
    top=[(float(le+x),naca_half(float(x),c,t)) for x in xs]
    bottom=[(xx,-zz) for xx,zz in reversed(top)]
    return top+bottom

def wire3d(points,axis,station):
    if axis=='H':  # xy plane horizontal tail; profile thickness is Z
        coords=[Vector(x,station,z) for x,z in points]
    else:         # vertical tail; profile thickness is Y
        coords=[Vector(x,z,station) for x,z in points]
    coords.append(coords[0]); return cq.Wire.makePolygon(coords)

def loft_sections(stations,kind,axis):
    wires=[]
    for station in stations:
        c,le=h_station(station) if axis=='H' else v_station(station)
        wire=wire3d(profile(c,le,0.10 if axis=='H' else 0.09,kind),axis,station)
        wires.append(wire)
    solid=cq.Solid.makeLoft(wires,ruled=True)
    assert solid.isValid(), (axis,kind)
    return solid

def export(shape,path):
    cq.exporters.export(shape,str(path),exportType='STEP')

def xy_bounds(shape):
    bb=shape.BoundingBox()
    return [round(bb.xmin,3),round(bb.ymin,3),round(bb.zmin,3),round(bb.xmax,3),round(bb.ymax,3),round(bb.zmax,3)]

# Airfoil envelopes: no fuselage, skins are solid envelopes, NOT printable solid blocks.
skins={}
for side,edge in [('L',-500),('R',500)]:
    skins[f'HS_{side}_Fixed']=loft_sections([0,edge],'fixed','H')
    rootsign=1 if edge>0 else -1
    skins[f'HS_{side}_Elevator']=loft_sections([rootsign*15,edge],'control','H')
skins['VS_Fin']=loft_sections([15,435],'fixed','V')
skins['VS_Rudder']=loft_sections([15,435],'control','V')
for name,shape in skins.items():
    export(shape,ROOT/'step/skins'/f'{name}.step')
export(cq.Compound.makeCompound(list(skins.values())),ROOT/'step'/'MG20_RevC_Airfoil_6Solids.step')

# Square cap rails: tapered 3D lofts, two per spar (top/bottom) on each half.
def cap_rect(c,le,t,cap_frac,width,height,top):
    xc=le+cap_frac*c
    zsurface=naca_half(cap_frac*c,c,t)-P['spar_undersize_mm']
    bottom=max(0.05,zsurface-height)
    if top:
        return [(xc-width/2,bottom),(xc+width/2,bottom),(xc+width/2,zsurface),(xc-width/2,zsurface)]
    return [(xc-width/2,-zsurface),(xc+width/2,-zsurface),(xc+width/2,-bottom),(xc-width/2,-bottom)]

structure={}
for axis,tag,limits in [('H','HS_L',(0,-500)),('H','HS_R',(0,500)),('V','VS',(15,435))]:
    for typ,fr,w,h in [('Main',.30,5,4),('Rear',.57,3,3)]:
        for top in (True,False):
            wires=[]
            for st in limits:
                c,le=h_station(st) if axis=='H' else v_station(st)
                coords=cap_rect(c,le,.10 if axis=='H' else .09,fr,w,h,top)
                wires.append(wire3d(coords,axis,st))
            shape=cq.Solid.makeLoft(wires,ruled=True)
            assert shape.isValid(),(tag,typ,top)
            structure[f'{tag}_{typ}_{"Top" if top else "Bottom"}_Cap']=shape

# Rib profiles with exact spar reliefs: remove upper/lower rectangles through outer skin.
def rib_poly(c,le,t,kind,notches=True):
    poly=Polygon(profile(c,le,t,kind,n=90)).buffer(0)
    if kind=='fixed' and notches:
        for fr,w,h in [(.30,5,4),(.57,3,3)]:
            xc=le+fr*c
            local_half=naca_half(fr*c,c,t)
            z_cap=local_half-P['spar_undersize_mm']
            seat=z_cap-h
            slot=w+P['laser_notch_extra_mm']
            # Use conservative full width cutter; slots in rib are slightly oversized.
            poly=poly.difference(box(xc-slot/2,seat,xc+slot/2,100))
            poly=poly.difference(box(xc-slot/2,-100,xc+slot/2,-seat))
    assert poly.geom_type=='Polygon' and poly.is_valid, (kind,c,poly.geom_type)
    return poly

rib_parts=[]
def add_rib(name,axis,station,kind,material):
    c,le=h_station(station) if axis=='H' else v_station(station)
    poly=rib_poly(c,le,.10 if axis=='H' else .09,kind)
    mat_thick={'ply2':2,'balsa3':3,'balsa2':2}[material]
    structure[f'Rib_{name}']=cq.Solid.extrudeLinear(wire3d(list(poly.exterior.coords)[:-1],axis,station-mat_thick/2),[],Vector(0,mat_thick,0) if axis=='H' else Vector(0,0,mat_thick))
    rib_parts.append((name,poly,material,axis,station))

# Leave ribs at y=0 and z=15 unmodeled until fuselage attachment geometry is known.
for s in [-485,-360,-240,-120,-15,15,120,240,360,485]:
    add_rib(f'HS_F_{s:+04d}','H',s,'fixed','ply2' if abs(s)==15 else 'balsa3')
for s in [-485,-385,-265,-145,-35,35,145,265,385,485]:
    add_rib(f'HS_E_{s:+04d}','H',s,'control','balsa2')
for z in [18,95,175,255,335,420]:
    add_rib(f'VS_F_{z:03d}','V',z,'fixed','ply2' if z==18 else 'balsa3')
for z in [18,95,175,255,335,420]:
    add_rib(f'VS_R_{z:03d}','V',z,'control','balsa2')
# Avoid generating an excessively unwieldy monolithic file: 32 ribs + 12 spars.
struct_compound=cq.Compound.makeCompound(list(structure.values()))
export(struct_compound,ROOT/'step'/'MG20_RevC_Skeleton.step')

# Laser templates, true millimetres, no machine-dependent kerf correction.
# One *instance* of every rib is present, not just labels for duplicated ribs.
for material in ['ply2','balsa3','balsa2']:
    doc=ezdxf.new('R2010')
    doc.header['$INSUNITS']=4  # mm
    doc.layers.new('CUT',dxfattribs={'color':1})
    doc.layers.new('ENGRAVE_ID',dxfattribs={'color':5})
    msp=doc.modelspace()
    cursor=10.0
    count=0
    for name,poly,mat,axis,station in rib_parts:
        if mat!=material: continue
        coords=list(poly.exterior.coords)[:-1]
        minx,miny,maxx,maxy=poly.bounds
        dx=10-minx
        dy=cursor-miny
        msp.add_lwpolyline([(x+dx,y+dy) for x,y in coords],close=True,dxfattribs={'layer':'CUT'})
        msp.add_text(name,dxfattribs={'layer':'ENGRAVE_ID','height':3,'insert':(10,cursor+maxy-miny+3)})
        cursor+=maxy-miny+15
        count+=1
    doc.saveas(ROOT/'cut'/f'Tail_Ribs_{material}_1to1.dxf')
    print(material, 'ribs',count,'layout length',round(cursor))

# 2D engineering plan: representative center root/tip profile and leading/hinge/trailing lines.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPoly
fig,axes=plt.subplots(2,1,figsize=(13,10),constrained_layout=True)
ax=axes[0]
for sign in [-1,1]:
    for kind in ['fixed','control']:
        for st in ([0,sign*500] if kind=='fixed' else [sign*15,sign*500]):
            c,le=h_station(st)
            aa,bb=ranges(c,kind)
            x0=le+aa; x1=le+bb
            ax.plot([st,st],[x0,x1],color='0.55',lw=.8)
        s0=0 if kind=='fixed' else sign*15
        pairs=[]
        for st in [s0,sign*500]:
            c,le=h_station(st)
            aa,bb=ranges(c,kind)
            pairs.append((st,le+aa,le+bb))
        ax.plot([p[0] for p in pairs],[p[1] for p in pairs], color='#0070A0' if kind=='fixed' else '#BA5D26')
        ax.plot([p[0] for p in pairs],[p[2] for p in pairs], color='#0070A0' if kind=='fixed' else '#BA5D26')
for st in [15,120,240,360,485,-15,-120,-240,-360,-485]:
    c,le=h_station(st); aa,bb=ranges(c,'fixed')
    ax.plot([st,st],[le+aa,le+bb],color='0.6',lw=.5)
ax.set_title('Horizontal tail — plan view (X aft, Y right), mm')
ax.set_xlabel('Y, mm');ax.set_ylabel('X, mm');ax.set_aspect('equal');ax.grid(alpha=.25)
ax=axes[1]
for kind in ['fixed','control']:
    st=[15,435]
    for cut in [0,1]:
        vals=[]
        for z in st:
            c,le=v_station(z); a,b=ranges(c,kind)
            vals.append(le+(a if cut==0 else b))
        ax.plot(vals,st,color='#0070A0' if kind=='fixed' else '#BA5D26')
    for z in st:
        c,le=v_station(z); a,b=ranges(c,kind)
        ax.plot([le+a,le+b],[z,z],color='0.55',lw=.8)
for z in [18,95,175,255,335,420]:
    c,le=v_station(z); a,b=ranges(c,'fixed')
    ax.plot([le+a,le+b],[z,z],color='0.6',lw=.5)
ax.set_title('Vertical tail — side view (base offset +15 mm), mm')
ax.set_xlabel('X, mm');ax.set_ylabel('Z, mm');ax.set_aspect('equal');ax.grid(alpha=.25)
fig.savefig(ROOT/'previews'/'MG20_Tail_RevC_geometry.png',dpi=180)
plt.close(fig)

# Two single-solid alternative STEP schemas for KOMPAS diagnostics.
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
for schema in ('AP203','AP214IS'):
    assert Interface_Static.SetCVal_s('write.step.schema',schema)
    writer=STEPControl_Writer()
    writer.Transfer(skins['HS_R_Fixed'].wrapped,STEPControl_AsIs)
    writer.Write(str(ROOT/'step'/f'HS_R_Fixed_{schema}.step'))

# After export, reimport each STEP and assert solids, nonzero volume, valid BRep.
verification=[]
for filepath in sorted((ROOT/'step').rglob('*.step')):
    reader=STEPControl_Reader()
    status=reader.ReadFile(str(filepath)); roots=reader.TransferRoots(); s=reader.OneShape()
    shape=cq.Shape(s); solids=shape.Solids(); volume=sum(q.Volume() for q in solids)
    valid=BRepCheck_Analyzer(s).IsValid()
    verification.append({'file':str(filepath.relative_to(ROOT)),'solid_count':len(solids),'volume_mm3':round(volume,3),'valid_OCCT':bool(valid),'bounds_mm':xy_bounds(shape),'file_bytes':filepath.stat().st_size})
    assert len(solids)>0 and volume>0 and valid,(filepath,status,roots,len(solids),volume,valid)
assert next(r for r in verification if r['file'].endswith('6Solids.step'))['solid_count']==6
assert next(r for r in verification if r['file'].endswith('Skeleton.step'))['solid_count']==len(structure)
(ROOT/'verification.json').write_text(json.dumps({'verification':'OpenCascade STEP reimport; KOMPAS GUI not available','files':verification},indent=2),encoding='utf-8')
print('STEP files',len(verification), 'skin parts',len(skins),'structural parts',len(structure),'ribs',len(rib_parts))
for item in verification: print(item['file'],item['solid_count'],'volume',item['volume_mm3'],'valid',item['valid_OCCT'])
