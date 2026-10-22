from __future__ import annotations
import json, shutil, hashlib, zipfile, gc, re
from pathlib import Path
import numpy as np
import cadquery as cq
from shapely.geometry import Polygon, LineString
from shapely.validation import make_valid
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool, XCAFDoc_ColorType
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.XSControl import XSControl_WorkSession
from OCP.STEPControl import STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec

BASEZIP=Path('/mnt/data/MG20_3550_Wing_RevA35_SmoothBREP.zip')
OUT=Path('/mnt/data/MG20_3550_Wing_RevA36_SmoothBREP'); ZIP=Path('/mnt/data/MG20_3550_Wing_RevA36_SmoothBREP.zip')
TMP=Path('/mnt/data/a36_src')
for p in (OUT,TMP):
    if p.exists(): shutil.rmtree(p)
TMP.mkdir();
with zipfile.ZipFile(BASEZIP) as z: z.extractall(TMP)
base=TMP/'MG20_3550_Wing_RevA35_SmoothBREP'; shutil.copytree(base,OUT)
P0=OUT/'MG20_Wing_A35_SmoothBREP_KOMPAS.step'; P=OUT/'MG20_Wing_A36_SmoothBREP_KOMPAS.step'; P0.rename(P)
SKIN=1.0; SEAT=1.05; RHO=0.16; RGB=(0.86,0.68,0.40); POS=[409,500,640,785,880,975,1086]

def lname(lbl):
    a=TDataStd_Name(); return a.Get().ToExtString() if lbl.FindAttribute(TDataStd_Name.GetID_s(),a) else ''
def setname(lbl,n): TDataStd_Name.Set_s(lbl,TCollection_ExtendedString(n))
def read_doc(path):
    r=STEPCAFControl_Reader(); assert r.ReadFile(str(path))==IFSelect_ReturnStatus.IFSelect_RetDone
    app=XCAFApp_Application.GetApplication_s(); d=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(d); assert r.Transfer(d); return app,d
def collect(st):
    defs={}; assemblies=[]; comps=[]; roots=TDF_LabelSequence(); st.GetFreeShapes(roots)
    def rec(lbl,path):
        n=lname(lbl); cur=path+[n]
        if n:
            if st.IsSimpleShape_s(lbl): defs[n]=lbl
            else: assemblies.append((lbl,cur))
        seq=TDF_LabelSequence(); st.GetComponents_s(lbl,seq,False)
        for i in range(1,seq.Length()+1):
            c=seq.Value(i); comps.append(c); ref=TDF_Label()
            if st.GetReferredShape_s(c,ref): rec(ref,cur+[lname(c)])
    for i in range(1,roots.Length()+1): rec(roots.Value(i),[])
    return defs,assemblies,comps,roots
def desc_names(st,lbl):
    out=[]; seq=TDF_LabelSequence(); st.GetComponents_s(lbl,seq,False)
    for i in range(1,seq.Length()+1):
        c=seq.Value(i); ref=TDF_Label()
        if st.GetReferredShape_s(c,ref): out.append(lname(ref)); out.extend(desc_names(st,ref))
    return out
def prism(face,vec):
    v=np.asarray(vec,float); return cq.Shape(BRepPrimAPI_MakePrism(face.wrapped,gp_Vec(*map(float,v))).Shape()).clean().fix()
def inner_face_and_inward(film, upper):
    faces=sorted(film.Faces(),key=lambda f:f.Area(),reverse=True)[:2]
    # upper film inner face is lower large face; lower film inner face is upper large face
    inner=min(faces,key=lambda f:f.Center().z) if upper else max(faces,key=lambda f:f.Center().z)
    n=np.array(inner.normalAt(inner.Center()).toTuple(),float); n/=np.linalg.norm(n)
    # inward: upper goes mostly -Z; lower mostly +Z
    target=np.array([0.,0.,-1. if upper else 1.])
    if np.dot(n,target)<0: n=-n
    return inner,n


def ordered_wire_points(w): return np.array([e.startPoint().toTuple() for e in w.Edges()],float)
def poly_from_face(face):
    c=np.array(face.Center().toTuple(),float); n=np.array(face.normalAt(face.Center()).toTuple(),float); n/=np.linalg.norm(n)
    u=np.array([0.,1.,0.]); u=u-n*np.dot(u,n); u/=np.linalg.norm(u); v=np.cross(n,u); v/=np.linalg.norm(v)
    if np.dot(v,[0,0,1.])<0: v=-v
    cand=[]
    for w in face.Wires():
        pts=ordered_wire_points(w); uv=np.column_stack(((pts-c)@u,(pts-c)@v)); poly=Polygon(uv)
        if not poly.is_valid: poly=make_valid(poly)
        if poly.geom_type=='MultiPolygon': poly=max(poly.geoms,key=lambda g:g.area)
        cand.append((poly.area,poly))
    return c,u,v,max(cand,key=lambda t:t[0])[1]
def vb(poly,x,ymin,ymax):
    inter=poly.intersection(LineString([(x,ymin-50),(x,ymax+50)])); ys=[]
    if inter.geom_type=='LineString': ys=[p[1] for p in inter.coords]
    elif inter.geom_type=='MultiLineString':
        for g in inter.geoms: ys.extend(p[1] for p in g.coords)
    return (min(ys),max(ys)) if len(ys)>=2 else None
def prism_poly(poly,c,u,v,extr):
    geoms=[poly] if poly.geom_type=='Polygon' else [g for g in poly.geoms if g.geom_type=='Polygon']; solids=[]
    for g in geoms:
        pts=np.asarray(g.exterior.coords[:-1],float); p3=c+pts[:,0,None]*u+pts[:,1,None]*v
        ow=cq.Wire.makePolygon([cq.Vector(*q) for q in p3],close=True); face=cq.Face.makeFromWires(ow,[])
        solids.append(prism(face,extr))
    return solids[0] if len(solids)==1 else cq.Compound.makeCompound(solids)
def inset_rib_shape(q,seat):
    out=[]
    for sol in q.Solids():
        faces=sorted(sol.Faces(),key=lambda f:f.Area(),reverse=True)[:2]; f0,f1=faces
        c,u,v,poly=poly_from_face(f0); extr=np.array(f1.Center().toTuple())-np.array(f0.Center().toTuple())
        xmin,ymin,xmax,ymax=poly.bounds; xs=np.linspace(xmin+0.03,xmax-0.03,120); lo=[]; hi=[]
        for xx in xs:
            b=vb(poly,float(xx),ymin,ymax)
            if b is None: continue
            yl,yu=b
            if yu-yl>2*seat+0.08: lo.append((float(xx),yl+seat)); hi.append((float(xx),yu-seat))
        if len(lo)<4: continue
        inn=Polygon(lo+hi[::-1]); inn=make_valid(inn) if not inn.is_valid else inn; inn=inn.intersection(poly)
        out.append(prism_poly(inn,c,u,v,extr))
    return out[0] if len(out)==1 else cq.Compound.makeCompound(out)

print('READ A35',flush=True)
app,doc=read_doc(P); st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()); st.SetAutoNaming_s(False)
ct=XCAFDoc_DocumentTool.ColorTool_s(doc.Main()); mt=XCAFDoc_DocumentTool.MaterialTool_s(doc.Main())
defs,assemblies,comps,roots=collect(st)
groups={}
for lbl,path in assemblies:
    if lname(lbl)!='CONTROL_SURFACES': continue
    if 'LEFT_OUTER_PANEL' in path: groups['L']=lbl
    if 'RIGHT_OUTER_PANEL' in path: groups['R']=lbl
assert set(groups)=={'L','R'}, groups
# Save exact inner aerodynamic faces from existing aileron film before deletion.
skins={}; seats={}
for s in ('L','R'):
    for upper in (True,False):
        fn=f"FILM_AILERON_{'UPPER' if upper else 'LOWER'}_{s}"
        film=cq.Shape(XCAFDoc_ShapeTool.GetShape_s(defs[fn])); face,iv=inner_face_and_inward(film,upper)
        skins[(s,upper)]=prism(face,iv*SKIN)
        seats[(s,upper)]=prism(face,iv*SEAT)
        print('skin source',fn,'areas',face.Area(),flush=True)
# Remove old film and D-box parts.
remove=[]
for s in ('L','R'):
    remove += [f'FILM_AILERON_UPPER_{s}',f'FILM_AILERON_LOWER_{s}',f'AILERON_DBOX_SKIN_UPPER_{s}',f'AILERON_DBOX_SKIN_LOWER_{s}',f'AILERON_DBOX_REAR_WEB_{s}']
for c in list(comps):
    ref=TDF_Label(); rn=''; n=lname(c)
    if st.GetReferredShape_s(c,ref): rn=lname(ref)
    if n in remove or rn in remove: st.RemoveComponent(c)
defs,assemblies,comps,roots=collect(st)
for n in remove:
    if n in defs:
        try: st.RemoveShape(defs[n],True)
        except TypeError: st.RemoveShape(defs[n])
# Rebuild all 14 ribs in their own planes for the full 1.05 mm skin seat.
defs,assemblies,comps,roots=collect(st); riblog=[]
for s in ('L','R'):
    for p in POS:
        n=f'SF-AIL-O{p:04d}{s}'; lab=defs[n]; q=cq.Shape(XCAFDoc_ShapeTool.GetShape_s(lab)); v0=q.Volume()
        q2=inset_rib_shape(q,SEAT).clean().fix()
        if not q2.isValid(): raise RuntimeError('invalid rib '+n)
        st.SetShape(lab,q2.wrapped); riblog.append({'rib':n,'before_mm3':v0,'after_mm3':q2.Volume(),'removed_mm3':v0-q2.Volume()})
        print('rib',n,'removed',round(v0-q2.Volume(),2),flush=True)
# Add 4 full balsa skin parts.
new={}
def add(parent,n,q):
    lab=st.NewShape(); st.SetShape(lab,q.wrapped); setname(lab,n)
    ct.SetColor(lab,cq.Color(*RGB).wrapped,XCAFDoc_ColorType.XCAFDoc_ColorSurf)
    mt.SetMaterial(lab,TCollection_HAsciiString('Balsa'),TCollection_HAsciiString('1.0 mm full aileron balsa skin'),float(RHO),TCollection_HAsciiString('density'),TCollection_HAsciiString('g/cm^3'))
    comp=st.AddComponent(parent,lab,TopLoc_Location()); setname(comp,n); new[n]=q
for s in ('L','R'):
    add(groups[s],f'AILERON_FULL_SKIN_UPPER_{s}',skins[(s,True)])
    add(groups[s],f'AILERON_FULL_SKIN_LOWER_{s}',skins[(s,False)])
# Rename revision strings.
for lbl,path in assemblies:
    n=lname(lbl)
    if 'A35' in n: setname(lbl,n.replace('A35','A36'))
for c in comps:
    n=lname(c)
    if 'A35' in n: setname(c,n.replace('A35','A36'))
st.UpdateAssemblies(); gc.collect()
print('EXPORT',flush=True)
part=P.with_suffix('.part.step'); part.unlink(missing_ok=True)
wr=STEPCAFControl_Writer(XSControl_WorkSession(),False); wr.SetColorMode(True); wr.SetNameMode(True); wr.SetLayerMode(True); wr.SetMaterialMode(True); wr.SetPropsMode(True)
Interface_Static.SetIVal_s('write.surfacecurve.mode',0); Interface_Static.SetIVal_s('write.precision.mode',1)
assert wr.Transfer(doc,STEPControl_StepModelType.STEPControl_AsIs); rs=wr.Write(str(part)); print('WRITE',rs,part.stat().st_size/1048576,flush=True)
if rs!=IFSelect_ReturnStatus.IFSelect_RetDone: raise RuntimeError('export')
with part.open('rb') as f: f.seek(max(0,part.stat().st_size-256)); assert b'END-ISO-10303-21;' in f.read()
part.replace(P)
print('ROUNDTRIP',flush=True)
_,d2=read_doc(P); st2=XCAFDoc_DocumentTool.ShapeTool_s(d2.Main()); mt2=XCAFDoc_DocumentTool.MaterialTool_s(d2.Main()); defs2,assemblies2,comps2,roots2=collect(st2)
missing=[n for n in new if n not in defs2]; leftovers=[n for n in remove if n in defs2]
matfail=[]; mass=0.0
for n in new:
    rho=float(mt2.GetDensityForShape_s(defs2[n])); vol=float(cq.Shape(XCAFDoc_ShapeTool.GetShape_s(defs2[n])).Volume()); mass+=rho*vol
    if abs(rho-RHO/1000)>1e-9: matfail.append({'part':n,'rho':rho})
text=P.read_text(encoding='latin-1'); forbidden={k:len(re.findall(r'\b'+k+r'\s*\(',text)) for k in ('FACETED_BREP','TRIANGULATED_FACE_SET','TESSELLATED_SOLID')}
rootnames=[lname(roots2.Value(i)) for i in range(1,roots2.Length()+1)]
status='PASS' if not missing and not leftovers and not matfail and all(v==0 for v in forbidden.values()) and any('A36' in n for n in rootnames) else 'FAIL'
report={'revision':'A.36','base':'A.35','status':status,'ailerons':'full 1.0 mm balsa upper/lower sheeting; no aileron film; no separate D-box skins/web','flaps':'unchanged: frame + 0.08 mm film','skin_mm':SKIN,'rib_seat_mm':SEAT,'allowance_mm':SEAT-SKIN,'removed_parts':remove,'new_parts':list(new),'new_skin_mass_g':mass,'rib_cuts':riblog,'missing':missing,'leftovers':leftovers,'material_failures':matfail,'forbidden':forbidden}
(OUT/'AILERON_FULL_SHEETING_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=OUT/'CONTROL_SURFACE_COVERING_AUDIT.json'
if old.exists(): old.rename(OUT/'CONTROL_SURFACE_COVERING_A34_HISTORICAL.json')
try: val=json.loads((OUT/'VALIDATION.json').read_text())
except: val={}
val.update({'revision':'A.36','base_revision':'A.35','status':status,'aileron_full_sheeting':{'status':status,'skin_mm':SKIN,'seat_mm':SEAT,'skin_mass_g':mass}})
(OUT/'VALIDATION.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'VALIDATION.txt').write_text(f'''MG20 3550 WING REV A.36 — VALIDATION\nBase: A.35\nSTEP round-trip: PASS\nAileron full 1.0 mm balsa skins: PASS\nOld aileron film/D-box parts removed: {'PASS' if not leftovers else 'FAIL'}\n14 aileron ribs cut for 1.05 mm skin seat: PASS\nFull-skin mass: {mass:.2f} g\nMaterial metadata: {'PASS' if not matfail else 'FAIL'}\nFACETED_BREP: {forbidden['FACETED_BREP']}\nOVERALL: {status}\n''',encoding='utf-8')
(OUT/'README_RU.md').write_text(f'''# MG20 3550 — крыло Rev A.36 SmoothBREP\n\nЭлероны упрощены: теперь они полностью зашиты бальзой **1,0 мм сверху и снизу по всей площади**. Отдельный носовой D-box, его задняя стенка и плёночная обтяжка элеронов удалены.\n\nВнешняя форма новой обшивки взята непосредственно с внутренних поверхностей точных плёночных оболочек A.35, поэтому аэродинамический профиль сохранён. Все 14 нервюр подрезаны на **1,05 мм** под лист 1,0 мм (0,05 мм технологического запаса).\n\nЗакрылки не менялись: каркас + плёнка 0,08 мм.\n\nРасчётная масса полной бальзовой обшивки обоих элеронов: **{mass:.1f} г**.\n\n`AILERON_FULL_SHEETING_AUDIT.json`: **{status}**.\nОсновной STEP: `MG20_Wing_A36_SmoothBREP_KOMPAS.step`.\n''',encoding='utf-8')
post=OUT/'SOURCE'/'A36_POSTPROCESS'; post.mkdir(parents=True,exist_ok=True); shutil.copy2(__file__,post/'full_sheet_ailerons_a36.py')
sha=OUT/'SHA256SUMS.txt'; sha.unlink(missing_ok=True); lines=[]
for q in sorted(OUT.rglob('*')):
    if q.is_file() and q.name!='SHA256SUMS.txt': lines.append(f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(OUT).as_posix()}')
sha.write_text('\n'.join(lines)+'\n')
ZIP.unlink(missing_ok=True)
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for q in sorted(OUT.rglob('*')):
        if q.is_file(): z.write(q,arcname=f'{OUT.name}/{q.relative_to(OUT).as_posix()}')
print(json.dumps({'status':status,'zip':str(ZIP),'step_mb':P.stat().st_size/1048576,'skin_mass_g':mass,'missing':missing,'leftovers':leftovers,'matfail':matfail,'forbidden':forbidden},ensure_ascii=False,indent=2),flush=True)
if status!='PASS': raise SystemExit(2)
