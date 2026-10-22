from __future__ import annotations
import json, math, os, shutil, time
from pathlib import Path
import cadquery as cq
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString, TCollection_HAsciiString
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.Quantity import Quantity_Color
from OCP.XSControl import XSControl_WorkSession
from OCP.STEPControl import STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus

P=Path('/mnt/data/MG20_3550_Wing_RevA32_SmoothBREP/MG20_Wing_A32_SmoothBREP_KOMPAS.step')
TMP=P.with_suffix('.with_materials.step')
print('read',flush=True)
r=STEPCAFControl_Reader(); assert r.ReadFile(str(P))==IFSelect_ReturnStatus.IFSelect_RetDone
app=XCAFApp_Application.GetApplication_s(); doc=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(doc); assert r.Transfer(doc)
st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()); ct=XCAFDoc_DocumentTool.ColorTool_s(doc.Main()); mt=XCAFDoc_DocumentTool.MaterialTool_s(doc.Main())

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

# RGB palette used by the A31/A32 smooth-BREP generator.
palette={
 'skin':((0.58,0.76,0.92),'Reference aerodynamic envelope',0.0,'Full aerodynamic reference solid; excluded from mass'),
 'control':((0.95,0.58,0.18),'Reference control-surface envelope',0.0,'Full control reference solid; excluded from mass'),
 'balsa':((0.86,0.68,0.40),'Balsa',0.16,'Design density 0.16 g/cm^3'),
 'ply':((0.55,0.31,0.13),'Birch plywood',0.70,'Design density 0.70 g/cm^3'),
 'spruce':((0.63,0.42,0.18),'Spruce / selected pine',0.45,'Design density 0.45 g/cm^3'),
 'pine':((0.70,0.49,0.24),'Spruce / selected pine',0.45,'Design density 0.45 g/cm^3'),
 'beech':((0.62,0.38,0.17),'Beech hardwood',0.72,'Design density 0.72 g/cm^3'),
 'web':((0.43,0.25,0.11),'Birch plywood web',0.70,'Design density 0.70 g/cm^3'),
 'aluminium':((0.70,0.73,0.76),'AD31T1 / 6060-T6 aluminium',2.70,'Design density 2.70 g/cm^3'),
 'steel':((0.42,0.45,0.48),'Steel',7.85,'Design density 7.85 g/cm^3'),
}

def _srgb_to_linear(c):
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
# cq.Color stores the generator's sRGB tuples as linear RGB inside XCAF.
for _k,_v in list(palette.items()):
    _rgb,_mn,_d,_desc=_v
    palette[_k]=(tuple(_srgb_to_linear(x) for x in _rgb),_mn,_d,_desc)

def rgb_for(lbl):
 c=Quantity_Color()
 if ct.GetColor_s(lbl,XCAFDoc_ColorType.XCAFDoc_ColorSurf,c):
  return (float(c.Red()),float(c.Green()),float(c.Blue()))
 return None

def classify(nm,lbl):
 # Explicit overrides for newly added A32 hardware and known functional solids.
 if nm.startswith(('FIXED_','CENTER_FLAP_','OUTER_FLAP_','AILERON_')) and nm in defs:
  # Do not override actual flap nose/hinge parts below; these have balsa color and
  # are caught by nearest palette. Only full aerodynamic/control bodies use exact colors.
  pass
 rgb=rgb_for(lbl)
 if rgb is None:
  if nm.startswith(('SERVO_BEZEL_','SERVO_BACKING_')):
   return ('ply','Birch plywood',0.70,'Design density 0.70 g/cm^3',None)
  if nm.startswith('WM_REAR_M6_BOLT_'):
   return ('steel','Steel',7.85,'Design density 7.85 g/cm^3',None)
  return ('unknown','Reference / unspecified',0.0,'No color/material source; excluded from mass',None)
 best=None
 for key,(ref,mname,dens,desc) in palette.items():
  dist=sum((rgb[i]-ref[i])**2 for i in range(3))**0.5
  if best is None or dist<best[0]: best=(dist,key,mname,dens,desc)
 if best[0]>0.08:
  return ('unknown','Reference / unspecified',0.0,'Unmapped STEP color; excluded from mass',rgb)
 return (best[1],best[2],best[3],best[4],rgb)

stats={}; assignments=[]
for nm,lbl in defs.items():
 key,mname,dens,desc,rgb=classify(nm,lbl)
 stats[key]=stats.get(key,0)+1
 mt.SetMaterial(lbl,TCollection_HAsciiString(mname),TCollection_HAsciiString(desc),float(dens),TCollection_HAsciiString('density'),TCollection_HAsciiString('g/cm^3'))
 assignments.append({'part':nm,'class':key,'material':mname,'density_g_cm3':dens,'color_rgb':rgb})
print('classes',stats,flush=True)

# Write with material/property modes enabled.
wr=STEPCAFControl_Writer(XSControl_WorkSession(),False); wr.SetColorMode(True); wr.SetNameMode(True); wr.SetLayerMode(True); wr.SetMaterialMode(True); wr.SetPropsMode(True)
Interface_Static.SetIVal_s('write.surfacecurve.mode',0); Interface_Static.SetIVal_s('write.precision.mode',1)
assert wr.Transfer(doc,STEPControl_StepModelType.STEPControl_AsIs)
assert wr.Write(str(TMP))==IFSelect_ReturnStatus.IFSelect_RetDone
TMP.replace(P)

# Verify material density survives STEP roundtrip on representative real parts.
r2=STEPCAFControl_Reader(); assert r2.ReadFile(str(P))==IFSelect_ReturnStatus.IFSelect_RetDone
d2=TDocStd_Document(TCollection_ExtendedString('XmlOcaf')); app.InitDocument(d2); assert r2.Transfer(d2)
st2=XCAFDoc_DocumentTool.ShapeTool_s(d2.Main()); mt2=XCAFDoc_DocumentTool.MaterialTool_s(d2.Main())
defs2={}
def walk2(lbl):
 nm=name(lbl)
 if st2.IsSimpleShape_s(lbl) and nm: defs2[nm]=lbl
 seq=TDF_LabelSequence(); st2.GetComponents_s(lbl,seq,False)
 for i in range(1,seq.Length()+1):
  c=seq.Value(i); ref=TDF_Label()
  if st2.GetReferredShape_s(c,ref): walk2(ref)
f2=TDF_LabelSequence(); st2.GetFreeShapes(f2)
for i in range(1,f2.Length()+1): walk2(f2.Value(i))
representatives={
 'balsa':'LEADING_EDGE_C_R',
 'ply':'RET_C595_DOUBLER_R',
 'spruce':'MAIN_CAP_TOP_1_R',
 'beech':'RET_COMPRESSION_SPACER_O000_O035_R',
 'aluminium':'MAIN_JOINER_20SQ_R',
 'steel':'RET_M4x55_BOLT_R',
}
verify={}
expected={'balsa':0.00016,'ply':0.00070,'spruce':0.00045,'beech':0.00072,'aluminium':0.00270,'steel':0.00785}
for key,nm in representatives.items():
 d=float(mt2.GetDensityForShape_s(defs2[nm])); verify[key]={'part':nm,'density_g_mm3':d,'expected_g_mm3':expected[key],'status':'PASS' if abs(d-expected[key])<1e-9 else 'FAIL'}
report={'revision':'A.32','status':'PASS' if all(v['status']=='PASS' for v in verify.values()) else 'FAIL','density_basis':'XCAF stores g/cm^3; GetDensityForShape returns g/mm^3','class_counts':stats,'representative_roundtrip':verify,'assignments':assignments}
(P.parent/'MATERIAL_METADATA_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('verify',verify,flush=True); print('STATUS',report['status'],flush=True)
if report['status']!='PASS': raise SystemExit(2)
