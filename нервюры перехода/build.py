from pathlib import Path
import sys, math, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.affinity import translate, affine_transform
import ezdxf

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import generate_wing_cutpack as w
STATIONS=[350,385,405,500,640]
THICKNESS=3.0
KERF=.20
GAP=.20
FEED=232.8
AXIS=8.013
originals={p.part_id:p for p in w.build_ribs()}

def cap_bounds(d):
    # Keep the external surfaces of the already dimensioned shoes.
    width,h,lo,up=w.normal_spar_geometry(595+d)
    return lo-h,up+h

def section(d):
    """Conservative filled cross-section of shoes, caps, rails and web box."""
    gy=595+d; cx=.35*w.chord_at_global_y(gy)
    parts=[]
    if d<=500:
        q=(d-230)/270
        endlo,endup=cap_bounds(500)
        low=AXIS-10.2+(endlo-(AXIS-10.2))*q
        high=AXIS+10.2+(endup-(AXIS+10.2))*q
        width=44.3 if d<=350 else 44.3+(w.spar_cap(1095)[0]-44.3)*(d-350)/150
        if d<=350:
            inner=10.1-5.1*(d-230)/120
            blo=AXIS-inner; tup=AXIS+inner
        else:
            g=w.joiner_transition_geometry('O',d)
            blo=g['lower_shoe_top']; tup=g['upper_shoe_bottom']
        # 2 mm transition walls; their outer surfaces follow the original taper.
        g=w.joiner_transition_geometry('O',d)
        webhalf=g['web_outer_half']-.05*(1-q)
        weblo=min(low,g['web_lower']); webhi=max(high,g['web_upper'])
        parts += [box(cx-width/2,low,cx+width/2,blo),box(cx-width/2,tup,cx+width/2,high),box(cx-webhalf,weblo,cx+webhalf,webhi)]
        if d<350:
            # Remaining thin rail tips at the inboard face of O0350.
            for sign in (-1,1):
                x0=cx+sign*17.15
                parts += [box(x0-5,AXIS+4.9,x0+5,AXIS+inner),box(x0-5,AXIS-inner,x0+5,AXIS-4.9)]
    else:
        low,high=cap_bounds(d)
        width=15.0; webhalf=8.5 # 15 mm cap and two external 1 mm walls
        weblo=low+5; webhi=high-5
        parts += [box(cx-7.5,low,cx+7.5,low+5),box(cx-7.5,high-5,cx+7.5,high),box(cx-webhalf,weblo,cx+webhalf,webhi)]
    if 460<=d<=540:
        # Existing 80 x 1 mm external web splice; allow 0.1 mm glue.
        endlo,endup=cap_bounds(500)
        zlo=weblo if d>500 else w.joiner_transition_geometry('O',d)['lower_rail_top']
        zhi=webhi if d>500 else w.joiner_transition_geometry('O',d)['upper_rail_bottom']
        parts += [box(cx-webhalf-1.1,zlo,cx+webhalf+1.1,zhi)]
    return unary_union(parts)

def projected_section(d,station):
    # Project neighbouring sections into the rib's coordinate plane, including twist.
    gy=595+d; ref=595+station
    le=.25*(310-w.chord_at_global_y(gy)); le0=.25*(310-w.chord_at_global_y(ref))
    angle=math.radians(w.twist_at_y(ref)-w.twist_at_y(gy))
    co,si=math.cos(angle),math.sin(angle)
    return affine_transform(section(d),[co,-si,si,co,co*(le-77.5)+77.5-le0,si*(le-77.5)])

def make_rib(d):
    gy=595+d; c=w.chord_at_global_y(gy); cx=.35*c
    outer,_,_=w.rib_outer_contour(gy,main_notch=False,rear_notch=True)
    if d==350: outer=w._clip_polygon_x(outer,w.fixed_hinge_front_x(c,'flap',gy),True)
    if d in (500,640): outer=w._clip_polygon_x(outer,w.fixed_hinge_front_x(c,'aileron',gy),True)
    air=Polygon(outer); assert air.is_valid
    samples=np.linspace(d-THICKNESS/2,d+THICKNESS/2,31)
    env=unary_union([projected_section(float(s),d) for s in samples])
    cut=env.buffer(GAP,join_style=2)
    # Open the top and bottom of the cutout: no fragile strips across the spar.
    minx,_,maxx,_=cut.bounds
    top=env.bounds[3]; bottom=env.bounds[1]
    cut=unary_union([cut,box(minx,top-GAP,maxx,100),box(minx,-100,maxx,bottom+GAP)])
    if d in (500,640):
        # Flat bearing edge against the external web / splice face; no tiny lips.
        cut=box(minx,-100,maxx,100)
    # Separate FWD/AFT; keep only the pieces connected to LE / rear region.
    body=air.difference(cut)
    candidates=list(body.geoms) if hasattr(body,'geoms') else [body]
    front=max([p for p in candidates if p.centroid.x<cx],key=lambda p:p.area)
    aft=max([p for p in candidates if p.centroid.x>cx],key=lambda p:p.area)
    out=[]
    for name,p in [('FWD',front),('AFT',aft)]:
        p=p.simplify(.002,preserve_topology=True)
        assert p.is_valid and p.geom_type=='Polygon' and not p.interiors
        assert p.intersection(env).area<1e-8
        # Dense independent station samples catch extrema between design samples.
        clearance=min(p.distance(projected_section(float(s),d)) for s in np.linspace(d-1.5,d+1.5,101))
        assert clearance>=.197, (d,name,clearance)
        out.append((f'O{d:04d}-{name}',p,clearance))
    return out,env,air

def dxf_file(path,entries):
    doc=ezdxf.new('R2010'); doc.units=4
    doc.layers.new('CUT',dxfattribs={'color':1}); m=doc.modelspace()
    for label,p in entries:
        m.add_lwpolyline(list(p.exterior.coords)[:-1],close=True,dxfattribs={'layer':'CUT'})
    doc.saveas(path)

def nc_file(path,entries):
    lines=['; MG20 TRANSITION RIBS 15x5 RevA; ONE CONSOLE',
      '; Birch plywood 3mm; kerf 0.20mm ALREADY compensated',
      '; GRBL $30=1000 $32=1; 1 pass; F232.8; M4 S1000',
      '; Origin bottom left; millimetres; no scaling or additional kerf',
      'G21','G90','G94','M5']
    for label,p in entries:
        tool=p.buffer(KERF/2,join_style=2)
        assert tool.is_valid and tool.geom_type=='Polygon' and not tool.interiors
        assert tool.buffer(-KERF/2,join_style=2).hausdorff_distance(p)<.015
        pts=[(round(x,3),round(y,3)) for x,y in tool.exterior.coords]
        assert Polygon(pts).is_valid
        lines += [f'; PART {label}',f'G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}','M4 S1000',f'G1 F{FEED:.1f}']
        lines += [f'G1 X{x:.3f} Y{y:.3f}' for x,y in pts[1:]]
        lines += ['M5']
    lines += ['M5','G0 X0 Y0','M5']
    path.write_text('\n'.join(lines)+'\n')

def svg_file(path,entries,W,H,labels=False):
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">']
    if labels:s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    for label,p in entries:
        pts=' '.join(f'{x:.3f},{H-y:.3f}' for x,y in p.exterior.coords)
        s.append(f'<polygon points="{pts}" fill="none" stroke="black" stroke-width="0.1"/>')
        if labels:
            q=p.representative_point();s.append(f'<text x="{q.x-10:.2f}" y="{H-q.y:.2f}" font-size="2.5">{label}</text>')
    s.append('</svg>');path.write_text('\n'.join(s))

def main():
    for folder in ['DXF_NOMINAL','SVG_NOMINAL','NC']: (ROOT/folder).mkdir(exist_ok=True)
    entries=[]; audit=[]; station_info=[]; sections=[]
    for i,d in enumerate(STATIONS):
        pieces,env,air=make_rib(d); sections.append((d,pieces,env,air))
        # Keep the relative front/aft placement of each complete rib on its row.
        shiftx=10-air.bounds[0]; shifty=12+i*48-air.bounds[1]
        for name,p,clearance in pieces:
            placed=translate(p,shiftx,shifty); entries.append((name,placed))
            dxf_file(ROOT/'DXF_NOMINAL'/f'{name}.dxf',[(name,p)])
            audit.append({'part':name,'station_mm':d,'thickness_mm':3,'area_mm2':p.area,'min_clearance_mm':clearance,'nominal_bounds':p.bounds})
        station_info.append({'station_mm':d,'parts':2,'material':'birch plywood','thickness_mm':3,'spar_envelope_xmin':env.bounds[0],'spar_envelope_xmax':env.bounds[2]})
    for i,(_,p) in enumerate(entries):
        assert box(0,0,297,420).contains(p.buffer(.1,join_style=2))
        for _,q in entries[:i]: assert p.distance(q)>2
    nc_file(ROOT/'NC'/'01_RIBS_ONE_CONSOLE_PLY3.nc',entries)
    dxf_file(ROOT/'DXF_NOMINAL'/'SHEET_ONE_CONSOLE_297x420.dxf',entries)
    svg_file(ROOT/'SVG_NOMINAL'/'SHEET_ONE_CONSOLE_297x420.svg',entries,297,420)
    svg_file(ROOT/'LAYOUT_LABELS_DO_NOT_CUT.svg',entries,297,420,True)
    # Full-scale narrow fit coupons: actual mating edges, 12 mm backing.
    tests=[]; yy=10
    for d,pieces,env,air in sections:
        if d not in (350,500,640):continue
        xx=10
        for name,p,_ in pieces:
            if name.endswith('FWD'): clipped=p.intersection(box(p.bounds[2]-15,-100,p.bounds[2]+1,100))
            else: clipped=p.intersection(box(p.bounds[0]-1,-100,p.bounds[0]+15,100))
            if clipped.geom_type!='Polygon':clipped=max(clipped.geoms,key=lambda a:a.area)
            q=translate(clipped,xx-clipped.bounds[0],yy-clipped.bounds[1]);tests.append((name+'-FIT',q));xx+=40
        yy+=45
    tests.append(('30x20_CALIBRATION',box(100,10,130,30)))
    nc_file(ROOT/'NC'/'00_FIT_TESTS_PLY3.nc',tests)
    dxf_file(ROOT/'DXF_NOMINAL'/'FIT_TESTS.dxf',tests)
    svg_file(ROOT/'FIT_TESTS_LABELS_DO_NOT_CUT.svg',tests,150,155,True)
    fig,axs=plt.subplots(5,1,figsize=(13,10),constrained_layout=True)
    for ax,(d,pieces,env,air) in zip(axs,sections):
        ax.fill(*env.exterior.xy,color='#ddbd84',alpha=.8,label='Spar envelope')
        for name,p,_ in pieces:
            ax.fill(*p.exterior.xy,color='#c8e3ef');ax.plot(*p.exterior.xy,color='#20506d',lw=.7)
        ax.set_title(f'O{d:04d} | FWD (left) + AFT (right) | mm',fontsize=10)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlim(0,240)
    fig.savefig(ROOT/'ASSEMBLY_SECTIONS.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9));
    for label,p in entries:
        ax.plot(*p.exterior.xy,lw=.7);q=p.representative_point();ax.text(q.x-10,q.y,label,fontsize=6)
    ax.set_aspect('equal');ax.set_xlim(0,297);ax.set_ylim(0,270);ax.grid(alpha=.2)
    ax.set_title('ONE CONSOLE | birch plywood 3 mm | origin bottom left')
    fig.savefig(ROOT/'LAYOUT.png',dpi=160);plt.close(fig)
    mass=sum(a['area_mm2'] for a in audit)*3*.0007
    result={'status':'GEOMETRY_PASS','structural_strength_verified':False,'station_count':5,'pieces_per_console':10,'stock_mm':[297,420,3],'kerf_mm':.2,'feed_mm_min':FEED,'power_S':1000,'passes':1,'nominal_mass_one_console_g':mass,'rib_details':audit,'station_details':station_info}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rib_details','station_details']},indent=2))

if __name__=='__main__':main()
