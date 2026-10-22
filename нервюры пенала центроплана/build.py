from pathlib import Path
import sys,re,json,zipfile,shutil
import numpy as np,ezdxf
from shapely.geometry import Polygon,box
from shapely.affinity import translate
import matplotlib.pyplot as plt
B=Path('/workspace/scratch/54d90fd0af06');S=B/'recovered/extracted';O=B/'output/center_penal_A26';O.mkdir(exist_ok=True)
sys.path.insert(0,str(S/'MG20_3550_Wing_RevA26_SmoothBREP/SOURCE'))
import generate_wing_cutpack as w
parts={p.part_id:p for p in w.build_ribs()}
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
interfaces={}
for t,file,d,offset in [(2,'RIBS_PLY2_ONE_CONSOLE_REVD.nc',35,0),(3,'HALF_RIBS_O0230_PLY3_REVC.nc',230,0)]:
 for i,side in enumerate(['FWD','AFT']):
  old=Polygon([(x,y) for x,y,*_ in list(ezdxf.readfile(S/f'DXF/O{d:04d}-{side}.dxf').modelspace().query('LWPOLYLINE'))[0].get_points()]);k=.25 if t==2 else .2
  p=paths(file)[i].buffer(-k/2,join_style=2)
  dx=old.bounds[0]-p.bounds[0] if side=='FWD' else old.bounds[2]-p.bounds[2]
  p=translate(p,dx,old.bounds[1]-p.bounds[1]);cx=.35*w.chord_at_global_y(595+d)
  tip=max(x for x,y in p.exterior.coords) if side=='FWD' else min(x for x,y in p.exterior.coords)
  ends=[y for x,y in p.exterior.coords if abs(x-tip)<1e-5]
  # Long vertical shoulder edges are the sleeve-facing boundary on both sides of tongue.
  edges=[(a,b) for a,b in zip(list(p.exterior.coords),list(p.exterior.coords)[1:]) if abs(a[0]-b[0])<1e-5 and abs(a[1]-b[1])>5 and abs(a[0]-tip)>5 and abs(a[0]-cx)<24]
  shoulder=min(edges,key=lambda ab:abs(abs(ab[0][0]-cx)-22.35))[0][0]
  interfaces[t,side]=(tip-cx,shoulder-cx,min(ends)-8.013,max(ends)-8.013)
print('interfaces',interfaces)
shapes={};checks=[]
root=parts['C595R'];shapes['C595']=(Polygon(root.outer,root.holes),3)
for station,t in [(560,2),(525,2),(445,3)]:
 p=parts[f'C{station}R'];outer=Polygon(p.outer);cx=.35*w.chord_at_global_y(station);axis=w.joiner_axis_z(station,'C')
 for side in ['FWD','AFT']:
  tip,shoulder,lo,hi=interfaces[t,side];tip+=cx;shoulder+=cx
  half=outer.intersection(box(-100,-100,shoulder,100) if side=='FWD' else box(shoulder,-100,400,100))
  tongue=box(min(tip,shoulder),axis+lo,max(tip,shoulder),axis+hi)
  shape=half.union(tongue);assert shape.is_valid and shape.geom_type=='Polygon' and not shape.interiors
  assert outer.buffer(.002).covers(shape)
  shapes[f'C{station}-{side}']=(shape,t)
  checks.append({'part':f'C{station}-{side}','axis':axis,'tongue_projection_nominal':abs(tip-shoulder),'tongue_height_nominal':hi-lo})
def write(name,items,stock):
 lines=[f'; MG20 CENTER PENAL {name}',f'; stock {stock}; ONE PENAL; one pass; grain along X','; M4 S1000; $30=1000 $32=1; overrides 100 percent','; kerf already applied; origin lower left; no scaling','G21','G90','G94','M5'];result=[]
 for label,x,y in items:
  p,t=shapes[label];k=.25 if t==2 else .2;p=translate(p,x-p.bounds[0],y-p.bounds[1]);assert all(p.distance(q)>3 for q in result);result.append(p)
  tool=p.buffer(k/2,join_style=2);assert tool.is_valid and tool.buffer(-k/2,join_style=2).hausdorff_distance(p)<.003
  for ring in [*tool.interiors,tool.exterior]:
   pts=[(round(a,3),round(b,3)) for a,b in ring.coords];assert pts[0]==pts[-1] and Polygon(pts).is_valid
   lines += [f'; {label}; PLY {t}mm',f'G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}','M4 S1000',f'G1 F{480 if t==2 else 232.8}']+[f'G1 X{a:.3f} Y{b:.3f}' for a,b in pts[1:]]+['M5']
 lines+=['M5','G0 X0 Y0','M5'];(O/(name+'.nc')).write_text('\n'.join(lines)+'\n')
 doc=ezdxf.new();doc.units=4
 for p in result:
  for ring in [p.exterior,*p.interiors]:doc.modelspace().add_lwpolyline(list(ring.coords)[:-1],close=True)
 doc.saveas(O/(name+'_nominal.dxf'))
 return result
rows=[write('01_C595_ROOT_PLY3',[('C595',10,12)],'280x60 mm'),write('02_C560_C525_PLY2',[('C560-FWD',10,12),('C560-AFT',110,12),('C525-FWD',10,67),('C525-AFT',110,67)],'290x120 mm'),write('03_C445_PLY3',[('C445-FWD',10,12),('C445-AFT',110,12)],'210x65 mm')]
fig,axs=plt.subplots(3,1,figsize=(13,9))
for ax,group in zip(axs,rows):
 for p in group:
  ax.plot(*p.exterior.xy)
  for h in p.interiors:ax.plot(*h.xy)
 ax.set_aspect('equal');ax.grid()
fig.tight_layout();fig.savefig(O/'layout.png',dpi=140)
(O/'checks.json').write_text(json.dumps(checks,indent=2))
for f in ['TEST_PLY2_TONGUE_REVD.nc','TEST_O0230_TONGUE_REVC.nc']:shutil.copy(S/f,O/f)
(O/'README_RU.txt').write_text('''Нервюры ОДНОГО пенала ЦЕНТРОПЛАНА MG20, адаптация A26.
C595 — цельная корневая, отверстие под трубу 20.4x20.4, авиационная берёзовая фанера 3 мм.
C560/C525 — передние и задние половинки, берёзовая фанера 2 мм.
C445 — передняя и задняя половинки, берёзовая фанера 3 мм.
Отсчёт от торца стыка: C595=0; C560=35; C525=70; C445=150 мм.
Комплект 7 деталей, для двух пеналов центроплана вырезать два комплекта.
Нервюры ставятся в плоскостях сечений крыла; не разворачивать все перпендикулярно наклонной трубе.
Учтён наклон оси трубы 2 градуса по проекту. Контуры и задние пазы A26, носовая обшивка 2 мм.
Посадки выступов перенесены из исправленных консольных NC RevD (2 мм) и RevC (3 мм).
Это предполагает одинаковый фактический зазор между рейками с консолью; сначала приложенные тесты.
TEST_O0230_TONGUE_REVC используется как тест посадки C445, несмотря на старое имя.
3 мм F232.8; 2 мм F480; M4 S1000; один проход; $30=1000; $32=1.
Керф уже внесён: 0.20/0.25 соответственно. Масштаб и overrides 100%. Ноль снизу слева. Волокна вдоль X.
Заготовки: 01 280x60; 02 290x120; 03 210x65 мм.
DXF — номинальные контуры БЕЗ компенсации; NC — готовые траектории с компенсацией.
Цельные C560/C525 не выбраны: минимальные перемычки вокруг необработанных крышек с зазором 0.2 составили бы около 0.86/1.24 мм.
C445: нижняя крышка 3 мм выступает за профиль примерно на 1.10 мм в худшей точке по ширине.
До вклейки проверить профилирование нижней крышки; для зазора 0.15 мм нужно снять в этой точке около 1.25 мм (остаток около 1.75 мм). Не снимать одинаково по всей ширине.
Профилирование крышек и прочность разрезных нервюр/пенала требуют отдельной проверки; NC не являются подтверждением прочности узла.
Переходные C365/C285/C205 в комплект не входят. Файлы для новых деталей, не для дорезки склеенного узла.
''')
with zipfile.ZipFile(B/'output/MG20_CENTER_PENAL_RIBS_A26.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,f.name)
print(checks)
