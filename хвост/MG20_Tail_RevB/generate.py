"""MG20 preliminary tail solids, millimetres. Regenerate: python generate.py"""
from pathlib import Path
import cadquery as cq
from OCP.STEPControl import STEPControl_Reader
OUT=Path(__file__).resolve().parent
# x aft, y right, z up. Generic simplified uncambered solid envelopes.
def horizontal(pts, thick=12):
    return cq.Workplane('XY').polyline(pts).close().extrude(thick/2,both=True).val()
def vertical(pts, thick=12):
    return cq.Workplane('XZ').polyline(pts).close().extrude(thick/2,both=True).val()
# Outline half span: 500 mm, chord 200 mm root, 120 mm tip.
# Fixed stabilizer ending at 70% chord, separate 30% elevator.
h_fixed = horizontal([(0,-0),(40,-500),(124,-500),(140,0),(124,500),(40,500)],16)
h_elev = horizontal([(140,-500),(160,-500),(200,0),(160,500),(124,500),(140,0)],10)
# Single vertical fin: height 420, root chord 300 and tip chord 180.
v_fixed = vertical([(0,0),(120,420),(246,420),(210,0)],16)
v_rudder = vertical([(210,0),(246,420),(300,420),(300,0)],10)
# non structural representation. Deliberate separate solids for CAD importing.
components={'H_Stabilizer':h_fixed,'H_Elevator':h_elev,'V_Fin':v_fixed,'V_Rudder':v_rudder}
for name,shape in components.items():
    cq.exporters.export(shape,str(OUT/(name+'.step')),exportType='STEP')
compound=cq.Compound.makeCompound(list(components.values()))
cq.exporters.export(compound,str(OUT/'MG20_Tail_RevB_4Solids.step'),exportType='STEP')
for fn in sorted(OUT.glob('*.step')):
    reader=STEPControl_Reader(); status=reader.ReadFile(str(fn)); num=reader.TransferRoots(); s=reader.OneShape()
    solids=len(cq.Shape(s).Solids())
    print(fn.name, 'status', status, 'roots',num,'solids',solids, 'bytes',fn.stat().st_size)
    assert solids == (4 if fn.name.startswith('MG20_') else 1), fn
