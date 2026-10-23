"""Supplementary concept cutting tolerance coupon and component/hinge schedules."""
from pathlib import Path
import csv,math
import ezdxf
from shapely.geometry import box
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cut'
OUT.mkdir(exist_ok=True)

def h(s):
    r=abs(s)/500.0
    return 200-80*r,40*r

def v(z):
    r=(z-15)/420.0
    return 300-120*r,120*r

# Test jig: test both real kerf and real stock sizes before batch cutting.
widths=[5.0,5.1,5.2,5.3,3.0,3.1,3.2,3.3]
poly=box(0,0,115,30)
for i,w in enumerate(widths):
    xc=9.5+13.5*i
    poly=poly.difference(box(xc-w/2,18,xc+w/2,31))
assert poly.geom_type=='Polygon' and poly.is_valid
doc=ezdxf.new('R2010')
doc.header['$INSUNITS']=4
doc.layers.new('CUT',dxfattribs={'color':1})
doc.layers.new('ENGRAVE_ID',dxfattribs={'color':5})
msp=doc.modelspace()
msp.add_lwpolyline(list(poly.exterior.coords)[:-1],close=True,dxfattribs={'layer':'CUT'})
for i,w in enumerate(widths):
    xc=9.5+13.5*i
    msp.add_text(f'{w:.1f}',dxfattribs={'layer':'ENGRAVE_ID','height':2.7,'insert':(xc-4,10)})
doc.saveas(OUT/'Fit_Test_Slots_5mm_3mm.dxf')

rib_positions={
'ply2': [('HS fixed root',s,'H') for s in [-15,15]]+[('VS fixed root',18,'V')],
'balsa3': [('HS fixed',s,'H') for s in [-485,-360,-240,-120,120,240,360,485]]+[('VS fixed',z,'V') for z in [95,175,255,335,420]],
'balsa2': [('HS elevator',s,'H') for s in [-485,-385,-265,-145,-35,35,145,265,385,485]]+[('VS rudder',z,'V') for z in [18,95,175,255,335,420]],
}
with (ROOT/'Cut_Parts_List.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['part','station_mm','axis','nominal_chord_mm','material','material_thickness_mm','quantity'])
    for material,items in rib_positions.items():
        for kind,st,axis in items:
            chord,le=h(st) if axis=='H' else v(st)
            w.writerow([kind,st,axis,round(chord,2),material,{'ply2':2,'balsa3':3,'balsa2':2}[material],1])
with (ROOT/'Spar_Caps_List.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['location','spar','surface','stock_mm','nominal_length_mm','qty','note'])
    for loc,span in [('HS left',500),('HS right',500),('VS',420)]:
        for row,stock in [('main','5 x 4'),('rear','3 x 3')]:
            for face in ['upper/right','lower/left']:
                w.writerow([loc,row,face,stock,span,1,'preliminary; strength and stock to be confirmed'])
with (ROOT/'Hinge_Stations_PRELIMINARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['surface','x_mm','y_mm','z_mm','hinge_axis','verification'])
    for s in [-455,-280,-95,95,280,455]:
        chord,le=h(s)
        w.writerow(['elevator',round(le+.7*chord,2),s,0,'spanwise Y','preliminary; check deflection and loads'])
    for z in [85,225,390]:
        chord,le=v(z)
        w.writerow(['rudder',round(le+.7*chord,2),0,z,'vertical Z','preliminary; check deflection and loads'])
print('Supplementary files created.')
