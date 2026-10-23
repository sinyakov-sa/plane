"""STEP export using the wing's XCAF workflow, without surface pcurves."""
import cadquery as cq
from OCP.XCAFApp import XCAFApp_Application
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.TopLoc import TopLoc_Location
from OCP.XSControl import XSControl_WorkSession
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus

def export_kompas(assembly, path):
    app=XCAFApp_Application.GetApplication_s()
    doc=TDocStd_Document(TCollection_ExtendedString('XmlOcaf'))
    app.InitDocument(doc)
    st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    st.SetAutoNaming_s(False)
    colors=XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    def name(label,s):TDataStd_Name.Set_s(label,TCollection_ExtendedString(s))
    root=st.NewShape();name(root,assembly.name)
    solids=[]
    for child in assembly.children:
        for i,solid in enumerate(child.obj.Solids()):
            n=child.name if len(child.obj.Solids())==1 else f'{child.name}_solid_{i+1:02d}'
            label=st.NewShape();st.SetShape(label,solid.wrapped);name(label,n)
            if child.color:colors.SetColor(label,child.color.wrapped,XCAFDoc_ColorType.XCAFDoc_ColorSurf)
            component=st.AddComponent(root,label,TopLoc_Location());name(component,n)
            solids.append(solid)
    st.UpdateAssemblies()
    writer=STEPCAFControl_Writer(XSControl_WorkSession(),False)
    writer.SetColorMode(True);writer.SetNameMode(True);writer.SetLayerMode(True)
    Interface_Static.SetCVal_s('write.step.schema','AP214IS')
    Interface_Static.SetIVal_s('write.step.assembly',1)
    Interface_Static.SetIVal_s('write.surfacecurve.mode',0)
    Interface_Static.SetIVal_s('write.precision.mode',1)
    if not writer.Transfer(doc,STEPControl_StepModelType.STEPControl_AsIs):raise RuntimeError('Transfer failed')
    if writer.Write(str(path))!=IFSelect_ReturnStatus.IFSelect_RetDone:raise RuntimeError('Write failed')
    # A single product containing all independent solids, without assembly references.
    flat=STEPControl_Writer()
    Interface_Static.SetIVal_s('write.step.assembly',0)
    Interface_Static.SetIVal_s('write.surfacecurve.mode',0)
    Interface_Static.SetIVal_s('write.precision.mode',1)
    if flat.Transfer(cq.Compound.makeCompound(solids).wrapped,STEPControl_StepModelType.STEPControl_AsIs)!=IFSelect_ReturnStatus.IFSelect_RetDone:raise RuntimeError('Flat transfer failed')
    if flat.Write(str(path).replace('.step','_FLAT.step'))!=IFSelect_ReturnStatus.IFSelect_RetDone:raise RuntimeError('Flat write failed')
