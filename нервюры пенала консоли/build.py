from pathlib import Path
import sys,re,json,shutil,zipfile
import ezdxf
from shapely.geometry import Polygon,box
from shapely.affinity import translate
import matplotlib.pyplot as plt
B=Path('/workspace/scratch/54d90fd0af06');S=B/'recovered/extracted';O=B/'output/console_A26'
sys.path.insert(0,str(S/'MG20_3550_Wing_RevA26_SmoothBREP/SOURCE'))
import generate_wing_cutpack as w
skin=w.skin_thickness
w.skin_thickness=lambda y:1.5 if abs(y-595)<1e-8 else skin(y)
new={p.part_id:p for p in w.build_ribs()}
def paths(file):
 out=[];cur=[]
 for l in (S/file).read_text().splitlines():
  if l.startswith('; PART_'):
   if cur:out.append(Polygon(cur))
   cur=[]
  m=re.match(r'G[01] X([-\d.]+) Y([-\d.]+)',l)
  if m and not l.startswith('G0 X0 Y0'):cur.append(tuple(map(float,m.groups())))
 if cur:out.append(Polygon(cur))
 return out
p2=paths('RIBS_PLY2_ONE_CONSOLE_REVD.nc');p3=paths('HALF_RIBS_O0230_PLY3_REVC.nc')
assert len(p2)==4 and len(p3)==2
shapes={};audit=[]
for d in (35,180,230):
 for i,side in enumerate(('FWD','AFT')):
  name=f'O{d:04d}-{side}';old=Polygon([(x,y) for x,y,*_ in list(ezdxf.readfile(S/f'DXF/{name}.dxf').modelspace().query('LWPOLYLINE'))[0].get_points()])
  k=.2 if d==230 else .25;tool=(p3[i] if d==230 else p2[(0 if d==35 else 2)+i]);nom=tool.buffer(-k/2,join_style=2)
  # Align on unmodified airfoil edges, never on the corrected tongue.
  dx=old.bounds[0]-nom.bounds[0] if side=='FWD' else old.bounds[2]-nom.bounds[2]
  dy=old.bounds[1]-nom.bounds[1];nom=translate(nom,dx,dy)
  if side=='FWD':shape=nom
  else:
   target=Polygon(new[f'O{d:04d}R'].outer);cut=125
   shape=nom.intersection(box(-100,-100,cut,100)).union(target.intersection(box(cut,-100,400,100)))
   assert shape.intersection(box(-100,-100,120,100)).symmetric_difference(nom.intersection(box(-100,-100,120,100))).area<1e-7
   assert shape.intersection(box(130,-100,400,100)).symmetric_difference(target.intersection(box(130,-100,400,100))).area<1e-7
  assert shape.is_valid and shape.geom_type=='Polygon'
  shapes[name]=(shape,k,2 if d!=230 else 3)
r=new['O0000R'];shapes['O0000']=(Polygon(r.outer,r.holes),.2,3)
def write(name,entries,stock):
 lines=[f'; MG20 {name}',f'; Stock {stock}; ONE PASS; grain along X','; G21 G90; origin lower left; overrides 100 percent','; Kerf already included; do not apply extra compensation','; GRBL $30=1000 $32=1','G21','G90','G94','M5'];placed=[]
 for label,shape,k,t,x,y in entries:
  shape=translate(shape,x-shape.bounds[0],y-shape.bounds[1]);assert all(shape.distance(q)>3 for q in placed);placed.append(shape)
  path=shape.buffer(k/2,join_style=2);assert path.is_valid
  assert path.buffer(-k/2,join_style=2).hausdorff_distance(shape)<.003
  for ring in list(path.interiors)+[path.exterior]:
   pts=[(round(a,3),round(b,3)) for a,b in ring.coords];assert pts[0]==pts[-1] and Polygon(pts).is_valid
   lines += [f'; {label}; ply {t}; kerf {k}',f'G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}','M4 S1000',f'G1 F{480 if t==2 else 232.8}']+[f'G1 X{a:.3f} Y{b:.3f}' for a,b in pts[1:]]+['M5']
  audit.append({'file':name,'part':label,'bounds':shape.bounds,'holes':len(shape.interiors),'valid':True})
 lines+=['M5','G0 X0 Y0','M5'];(O/(name+'.nc')).write_text('\n'.join(lines)+'\n')
 return placed
root=write('01_ROOT_O0000_PLY3_SKIN1p5',[('O0000',*shapes['O0000'],10,12)],'280x60 mm')
ply2=write('02_O0035_O0180_PLY2',[(n,*shapes[n],x,y) for n,x,y in [('O0035-FWD',10,12),('O0035-AFT',100,12),('O0180-FWD',10,62),('O0180-AFT',100,62)]],'270x110 mm')
ply3=write('03_O0230_PLY3',[(n,*shapes[n],x,12) for n,x in [('O0230-FWD',10),('O0230-AFT',95)]],'190x60 mm')
fig,axs=plt.subplots(3,1,figsize=(13,9))
for ax,group in zip(axs,[root,ply2,ply3]):
 for p in group:
  ax.plot(*p.exterior.xy)
  for h in p.interiors:ax.plot(*h.xy)
 ax.set_aspect('equal');ax.grid()
fig.tight_layout();fig.savefig(O/'layout.png',dpi=140)
(O/'verification.json').write_text(json.dumps(audit,indent=2))
for name in ['TEST_PLY2_TONGUE_REVD.nc','TEST_O0230_TONGUE_REVC.nc']:shutil.copy(S/name,O/name)
(O/'README_RU.txt').write_text('''MG20: комплект НОВЫХ нервюр пенала ОДНОЙ консоли, адаптация RevA26.
7 деталей. Для второй консоли повторить комплект, профиль не зеркалить.
01: O0000, берёзовая авиационная фанера 3 мм; 280x60 мм; F232.8.
02: передние и задние половинки O0035/O0180, берёзовая фанера 2 мм; 270x110 мм; F480.
03: половинки O0230, берёзовая фанера 3 мм; 190x60 мм; F232.8.
Везде M4 S1000, один проход, $30=1000, $32=1. Ноль снизу слева, волокна вдоль X.
Компенсация уже внесена: 2 мм — 0.25 мм; 3 мм — 0.20 мм. Масштаб и overrides 100%.
Посадки сохранены из RevD (2 мм) и эмпирической RevC (O0230).
Профиль хвостов и пазы заднего лонжерона — RevA26. O0035 теперь до полной задней кромки.
O0000 новая: носовой контур под обшивку 1.5 мм, отверстие 20.4x20.4 мм.
На новую O0000 полоски бальзы 0.5 мм НЕ нужны.
O0180/O0230 новые: длина уже по RevA26; компенсирующая накладка около 4 мм на заднюю рейку НЕ нужна.
Не использовать эти NC для дорезки установленных деталей.
Перед основными файлами используйте соответствующие тесты выступов; примерка без натяга.
Проверены замкнутость, отсутствие самопересечений/наложений и сохранение посадок.
Комплект не является подтверждением прочности разрезных нервюр или полной адаптацией переходных колодок и остального крыла.
''')
with zipfile.ZipFile(B/'output/MG20_CONSOLE_RIBS_A26_ADAPTED.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,f.name)
print(json.dumps(audit,indent=2))
