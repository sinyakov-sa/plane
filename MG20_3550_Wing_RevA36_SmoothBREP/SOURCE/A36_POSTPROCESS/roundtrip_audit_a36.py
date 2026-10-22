import json,re
from pathlib import Path
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence,TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_ReturnStatus
p=Path('/mnt/data/MG20_3550_Wing_RevA36_SmoothBREP/MG20_Wing_A36_SmoothBREP_KOMPAS.step')
r=STEPCAFControl_Reader(); print('read',r.ReadFile(str(p)),flush=True)
app=XCAFApp_Application.GetApplication_s(); d=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(d); print('transfer',r.Transfer(d),flush=True)
st=XCAFDoc_DocumentTool.ShapeTool_s(d.Main()); mt=XCAFDoc_DocumentTool.MaterialTool_s(d.Main())
def name(lbl):
 a=TDataStd_Name(); return a.Get().ToExtString() if lbl.FindAttribute(TDataStd_Name.GetID_s(),a) else ''
roots=TDF_LabelSequence(); st.GetFreeShapes(roots); defs={}; rootnames=[]
def rec(lbl):
 n=name(lbl)
 if n and st.IsSimpleShape_s(lbl): defs[n]=lbl
 seq=TDF_LabelSequence(); st.GetComponents_s(lbl,seq,False)
 for i in range(1,seq.Length()+1):
  c=seq.Value(i); ref=TDF_Label()
  if st.GetReferredShape_s(c,ref): rec(ref)
for i in range(1,roots.Length()+1): rootnames.append(name(roots.Value(i))); rec(roots.Value(i))
new=[f'AILERON_FULL_SKIN_{u}_{s}' for s in ('L','R') for u in ('UPPER','LOWER')]
old=[]
for s in ('L','R'): old += [f'FILM_AILERON_UPPER_{s}',f'FILM_AILERON_LOWER_{s}',f'AILERON_DBOX_SKIN_UPPER_{s}',f'AILERON_DBOX_SKIN_LOWER_{s}',f'AILERON_DBOX_REAR_WEB_{s}']
missing=[n for n in new if n not in defs]; leftovers=[n for n in old if n in defs]
rhos={n:float(mt.GetDensityForShape_s(defs[n])) for n in new if n in defs}
print(json.dumps({'rootnames':rootnames,'missing':missing,'leftovers':leftovers,'densities':rhos},ensure_ascii=False,indent=2),flush=True)
