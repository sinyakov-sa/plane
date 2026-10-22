from __future__ import annotations
import json, time
from pathlib import Path
import cadquery as cq
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence,TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_ReturnStatus
p=Path('/mnt/data/MG20_3550_Wing_RevA32_SmoothBREP/MG20_Wing_A32_SmoothBREP_KOMPAS.step')
print('reading',flush=True); t=time.time(); r=STEPCAFControl_Reader(); rs=r.ReadFile(str(p)); print('read',rs,'sec',time.time()-t,flush=True)
app=XCAFApp_Application.GetApplication_s(); doc=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(doc)
print('transfer',flush=True); t=time.time(); ok=bool(r.Transfer(doc)); print('transfer ok',ok,'sec',time.time()-t,flush=True)
st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
def name(lbl):
 a=TDataStd_Name();
 if lbl.FindAttribute(TDataStd_Name.GetID_s(),a): return a.Get().ToExtString()
 return ''
defs={}
def walk(lbl):
 nm=name(lbl)
 if st.IsSimpleShape_s(lbl) and nm: defs[nm]=lbl
 seq=TDF_LabelSequence(); st.GetComponents_s(lbl,seq,False)
 for i in range(1,seq.Length()+1):
  c=seq.Value(i); ref=TDF_Label()
  if st.GetReferredShape_s(c,ref): walk(ref)
free=TDF_LabelSequence(); st.GetFreeShapes(free)
for i in range(1,free.Length()+1): walk(free.Value(i))
print('free',free.Length(),'root',name(free.Value(1)),'defs',len(defs),flush=True)
required=['RET_COMPRESSION_SPACER_O000_O035_R','RET_COMPRESSION_SPACER_O000_O035_L','RET_M4_CAPTIVE_NUT_R','RET_M4_CAPTIVE_NUT_L','RET_M4_SOCKET_HEAD_R','RET_M4_SOCKET_HEAD_L']
print('required', {n:n in defs for n in required},flush=True)
def sh(n): return cq.Shape(st.GetShape_s(defs[n]))
def common(a,b):
 try:
  c=sh(a).intersect(sh(b)); return 0.0 if c.isNull() else float(c.Volume())
 except ValueError:return 0.0
checks=[]
for side in 'RL':
 for a,b in [
  (f'RET_C595_DOUBLER_{side}',f'C595{side}'),(f'RET_O000_DOUBLER_{side}',f'O0000{side}'),(f'RET_O035_GUIDE_{side}',f'O0035{side}'),
  (f'RET_COMPRESSION_SPACER_O000_O035_{side}',f'O0000{side}'),(f'RET_COMPRESSION_SPACER_O000_O035_{side}',f'O0035{side}'),
  (f'RET_M4x55_BOLT_{side}',f'C595{side}'),(f'RET_M4x55_BOLT_{side}',f'O0000{side}'),(f'RET_M4x55_BOLT_{side}',f'O0035{side}'),
  (f'RET_M4x55_BOLT_{side}',f'MAIN_JOINER_20SQ_{side}')]:
  v=common(a,b); checks.append({'a':a,'b':b,'common_volume_mm3':v,'status':'PASS' if v<1e-4 else 'FAIL'}); print(a,b,v,flush=True)
for side in 'RL':
 for b in [f'RET_C595_DOUBLER_{side}',f'RET_O000_DOUBLER_{side}',f'RET_O035_GUIDE_{side}',f'RET_COMPRESSION_SPACER_O000_O035_{side}',f'RET_C_NUT_BLOCK_{side}']:
  v=common(f'RET_M4x55_BOLT_{side}',b); checks.append({'a':f'RET_M4x55_BOLT_{side}','b':b,'common_volume_mm3':v,'status':'PASS' if v<1e-4 else 'FAIL'}); print('bore',side,b,v,flush=True)
audit={'revision':'A.32','status':'PASS' if ok and all(c['status']=='PASS' for c in checks) and all(n in defs for n in required) else 'FAIL','roundtrip_import':ok,'root_name':name(free.Value(1)),'required_parts':{n:n in defs for n in required},'checks':checks}
out=p.parent/'PANEL_RETENTION_AUDIT.json'; out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print('AUDIT',audit['status'],flush=True)
