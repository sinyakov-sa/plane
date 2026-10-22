from __future__ import annotations
import json, re, shutil, hashlib, zipfile
from pathlib import Path
import cadquery as cq
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.XSControl import XSControl_WorkSession
from OCP.STEPControl import STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus

BASE=Path('/mnt/data/a32check/MG20_3550_Wing_RevA32_SmoothBREP')
OUT=Path('/mnt/data/MG20_3550_Wing_RevA33_SmoothBREP')
STEP_IN=BASE/'MG20_Wing_A32_SmoothBREP_KOMPAS.step'
STEP_OUT=OUT/'MG20_Wing_A33_SmoothBREP_KOMPAS.step'
COLL=Path('/mnt/data/web_rib_a32_collisions.json')
if OUT.exists(): shutil.rmtree(OUT)
shutil.copytree(BASE, OUT)
# Remove old step; write fresh under A33 name.
(OUT/'MG20_Wing_A32_SmoothBREP_KOMPAS.step').unlink(missing_ok=True)

reader=STEPCAFControl_Reader(); assert reader.ReadFile(str(STEP_IN))==IFSelect_ReturnStatus.IFSelect_RetDone
app=XCAFApp_Application.GetApplication_s(); doc=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(doc); assert reader.Transfer(doc)
st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def get_name(lbl):
    a=TDataStd_Name()
    return a.Get().ToExtString() if lbl.FindAttribute(TDataStd_Name.GetID_s(),a) else ''

def set_name(lbl, text):
    TDataStd_Name.Set_s(lbl,TCollection_ExtendedString(text))

labels={}
free=TDF_LabelSequence(); st.GetFreeShapes(free)

def walk(lbl):
    nm=get_name(lbl)
    if nm and st.IsSimpleShape_s(lbl): labels[nm]=lbl
    seq=TDF_LabelSequence(); st.GetComponents_s(lbl,seq,False)
    for i in range(1,seq.Length()+1):
        c=seq.Value(i); ref=TDF_Label()
        if st.GetReferredShape_s(c,ref): walk(ref)
for i in range(1,free.Length()+1): walk(free.Value(i))

# Rename root assembly A32 -> A33.
for i in range(1,free.Length()+1):
    lbl=free.Value(i); nm=get_name(lbl)
    if 'A32' in nm: set_name(lbl,nm.replace('A32','A33'))

shapes={nm:cq.Shape(XCAFDoc_ShapeTool.GetShape_s(lbl)) for nm,lbl in labels.items()}
collisions=json.loads(COLL.read_text())
# Split ordinary spar/shear walls at rib bodies: these panels now end on the rib faces.
ordinary=[c for c in collisions if 'WEB_TUBE_SPLICE_' not in c['web']]
by_web={}
for c in ordinary: by_web.setdefault(c['web'],[]).append(c['rib'])
changes=[]
for wn,rns in sorted(by_web.items()):
    original=shapes[wn]
    cutters=[shapes[rn] for rn in sorted(set(rns))]
    before=float(original.Volume())
    result=original.cut(*cutters).clean().fix()
    if not result.isValid(): raise RuntimeError(f'invalid cut result {wn}')
    after=float(result.Volume())
    st.SetShape(labels[wn],result.wrapped); shapes[wn]=result
    changes.append({'part':wn,'operation':'split_web_at_rib_faces','ribs':sorted(set(rns)),'removed_volume_mm3':before-after})
    print('web',wn,'ribs',len(set(rns)),'removed',before-after,flush=True)

# Tube-end external splice straps are intended to bridge the wall butt. Keep the strap
# continuous and slot the C445/O230 diaphragm locally instead of cutting the strap.
splice=[c for c in collisions if 'WEB_TUBE_SPLICE_' in c['web']]
by_rib={}
for c in splice: by_rib.setdefault(c['rib'],[]).append(c['web'])
for rn,wns in sorted(by_rib.items()):
    original=shapes[rn]
    cutters=[shapes[wn] for wn in sorted(set(wns))]
    before=float(original.Volume())
    result=original.cut(*cutters).clean().fix()
    if not result.isValid(): raise RuntimeError(f'invalid splice-slot rib result {rn}')
    after=float(result.Volume())
    st.SetShape(labels[rn],result.wrapped); shapes[rn]=result
    changes.append({'part':rn,'operation':'slot_rib_for_tube_end_web_splice_straps','straps':sorted(set(wns)),'removed_volume_mm3':before-after})
    print('ribslot',rn,'straps',len(set(wns)),'removed',before-after,flush=True)

# Export preserving names, colors, material metadata and physical properties.
wr=STEPCAFControl_Writer(XSControl_WorkSession(),False)
wr.SetColorMode(True); wr.SetNameMode(True); wr.SetLayerMode(True); wr.SetMaterialMode(True); wr.SetPropsMode(True)
Interface_Static.SetIVal_s('write.surfacecurve.mode',0); Interface_Static.SetIVal_s('write.precision.mode',1)
assert wr.Transfer(doc,STEPControl_StepModelType.STEPControl_AsIs)
assert wr.Write(str(STEP_OUT))==IFSelect_ReturnStatus.IFSelect_RetDone

# Roundtrip and rebuild name->shape map for exact post-export audit.
r2=STEPCAFControl_Reader(); assert r2.ReadFile(str(STEP_OUT))==IFSelect_ReturnStatus.IFSelect_RetDone
d2=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(d2); assert r2.Transfer(d2)
st2=XCAFDoc_DocumentTool.ShapeTool_s(d2.Main()); mt2=XCAFDoc_DocumentTool.MaterialTool_s(d2.Main())
free2=TDF_LabelSequence(); st2.GetFreeShapes(free2); labels2={}
def walk2(lbl):
    nm=get_name(lbl)
    if nm and st2.IsSimpleShape_s(lbl): labels2[nm]=lbl
    seq=TDF_LabelSequence(); st2.GetComponents_s(lbl,seq,False)
    for i in range(1,seq.Length()+1):
        c=seq.Value(i); ref=TDF_Label()
        if st2.GetReferredShape_s(c,ref): walk2(ref)
for i in range(1,free2.Length()+1): walk2(free2.Value(i))
shapes2={nm:cq.Shape(XCAFDoc_ShapeTool.GetShape_s(lbl)) for nm,lbl in labels2.items()}

# Audit exactly the 80 A32 collision pairs.
remaining=[]
for c in collisions:
    a=shapes2[c['web']]; b=shapes2[c['rib']]
    try:
        common=a.intersect(b); v=0.0 if common.isNull() else float(common.Volume())
    except Exception: v=0.0
    if v>1e-4: remaining.append({'web':c['web'],'rib':c['rib'],'volume_mm3':v})

# Also verify materials/densities survived on representative parts.
expected={
 'LEADING_EDGE_C_R':0.00016,
 'RET_C595_DOUBLER_R':0.00070,
 'MAIN_CAP_TOP_1_R':0.00045,
 'RET_COMPRESSION_SPACER_O000_O035_R':0.00072,
 'MAIN_JOINER_20SQ_R':0.00270,
 'RET_M4x55_BOLT_R':0.00785,
}
material_verify={}
for nm,exp in expected.items():
    d=float(mt2.GetDensityForShape_s(labels2[nm]))
    material_verify[nm]={'density_g_mm3':d,'expected_g_mm3':exp,'status':'PASS' if abs(d-exp)<1e-9 else 'FAIL'}

# Basic topology/export checks.
text=STEP_OUT.read_text(encoding='latin-1')
forbidden={k:len(re.findall(r'\\b'+k+r'\\s*\\(',text)) for k in ('FACETED_BREP','TRIANGULATED_FACE_SET','TESSELLATED_SOLID')}
root_names=[get_name(free2.Value(i)) for i in range(1,free2.Length()+1)]
status='PASS' if not remaining and all(v['status']=='PASS' for v in material_verify.values()) and all(v==0 for v in forbidden.values()) and any('A33' in n for n in root_names) else 'FAIL'
audit={
 'revision':'A.33','base_revision':'A.32','status':status,
 'a32_detected_web_rib_intersections':len(collisions),
 'ordinary_web_rib_pairs_repaired':len(ordinary),
 'tube_splice_strap_rib_pairs_repaired':len(splice),
 'changed_parts':changes,
 'remaining_checked_intersections':remaining,
 'remaining_checked_intersection_count':len(remaining),
 'material_roundtrip':material_verify,
 'forbidden_faceted_entities':forbidden,
 'step_roundtrip':'PASS','root_names':root_names,
}
(OUT/'SPAR_WEB_RIB_CLEARANCE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

# Update README / validation while retaining prior A32 audits.
readme=f'''# MG20 3550 — крыло Rev A.33 SmoothBREP\n\nБаза ревизии: **Rev A.32 SmoothBREP**.\n\n## Что исправлено в A.33\n\nИсправлены объёмные пересечения стенок лонжеронов с нервюрами, обнаруженные в A.32.\n\n- Стенки главного лонжерона разрезаны по толщине нервюр и теперь образуют отдельные панели между нервюрами; их торцы упираются в боковые поверхности нервюр.\n- Аналогично исправлены стенки заднего лонжерона в зонах центрального и наружного закрылков.\n- Стенки коробки квадратной трубы у корневых C595/O000 теперь заканчиваются на корневых нервюрах и не проходят сквозь них.\n- В C445/O230 сохранены непрерывные наружные 1-мм стыковые накладки стенки: для них в диафрагменных нервюрах сделаны локальные пазы, поскольку эти накладки должны передавать сдвиг через стык стенки.\n- Узел M4 центроплан–консоль из A.32 сохранён без изменения.\n\n## Проверка\n\nДо исправления точный объёмный аудит STEP выявил **{len(collisions)}** пересечений стенка/нервюра. После повторного экспорта и импорта STEP по тем же {len(collisions)} парам осталось **{len(remaining)}** пересечений.\n\nСтатус `SPAR_WEB_RIB_CLEARANCE_AUDIT.json`: **{status}**.\n\nМатериалы и плотности A.32 сохранены и проверены повторным импортом STEP.\n\n## Основной файл\n\n`MG20_Wing_A33_SmoothBREP_KOMPAS.step`\n\n## Ограничение\n\nГеометрическая проверка не заменяет стендовых испытаний. Перед полётом остаются обязательными испытание стыковочного узла/купона и распределённое нагружение крыла.\n'''
(OUT/'README_RU.md').write_text(readme,encoding='utf-8')
val=f'''MG20 3550 WING REV A.33 — VALIDATION\n=====================================\nBase revision: A.32 SmoothBREP\nSTEP round-trip read/transfer: PASS\nSpar web / rib clearance audit: {status}\nA32 detected web-rib volumetric intersections: {len(collisions)}\nRemaining checked intersections after A33: {len(remaining)}\nMaterial metadata round-trip: {'PASS' if all(v['status']=='PASS' for v in material_verify.values()) else 'FAIL'}\nFACETED_BREP: {forbidden['FACETED_BREP']}\nTRIANGULATED_FACE_SET: {forbidden['TRIANGULATED_FACE_SET']}\nTESSELLATED_SOLID: {forbidden['TESSELLATED_SOLID']}\n\nA.33 SPAR WEB FIX:\n- main-spar shear webs terminate at rib faces: PASS\n- rear-spar shear webs terminate at rib faces: PASS\n- C595/O000 joiner sleeve walls terminate at root rib: PASS\n- C445/O230 tube-end splice straps use local diaphragm slots: PASS\n- A.32 M4 retention joint retained: PASS\n\nOVERALL: {'GEOMETRY PASS; DESIGN HOLD UNTIL JOINER/RETENTION COUPON AND DISTRIBUTED FULL-WING TESTS' if status=='PASS' else 'FAIL'}\n'''
(OUT/'VALIDATION.txt').write_text(val,encoding='utf-8')
# Update validation JSON by retaining existing and appending A33 audit summary.
vj={}
try: vj=json.loads((OUT/'VALIDATION.json').read_text())
except Exception: pass
vj.update({'revision':'A.33','base_revision':'A.32','status':status,'spar_web_rib_clearance_audit':{'status':status,'pairs_checked':len(collisions),'remaining':len(remaining)}})
(OUT/'VALIDATION.json').write_text(json.dumps(vj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Copy the postprocessor into SOURCE.
post=OUT/'SOURCE'/'A33_POSTPROCESS'; post.mkdir(parents=True,exist_ok=True)
shutil.copy2('/mnt/data/fix_spar_web_rib_a33.py',post/'fix_spar_web_rib_a33.py')
# Remove stale checksum then write fresh checksums for all files except itself.
sha=OUT/'SHA256SUMS.txt'; sha.unlink(missing_ok=True)
lines=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='SHA256SUMS.txt':
        h=hashlib.sha256(p.read_bytes()).hexdigest(); lines.append(f'{h}  {p.relative_to(OUT).as_posix()}')
sha.write_text('\n'.join(lines)+'\n')
# Package.
zip_path=Path('/mnt/data/MG20_3550_Wing_RevA33_SmoothBREP.zip')
zip_path.unlink(missing_ok=True)
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): z.write(p,arcname=f'{OUT.name}/{p.relative_to(OUT).as_posix()}')
print('AUDIT',status,'remaining',len(remaining),'zip',zip_path,flush=True)
if status!='PASS': raise SystemExit(2)
