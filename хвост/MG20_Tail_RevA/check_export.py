"""Compare STEP geometry with a pre-fix directory supplied on the command line."""
from pathlib import Path
import sys,json,re
import numpy as np
from scipy.optimize import linear_sum_assignment
import cadquery as cq
here=Path(__file__).resolve().parent
old=Path(sys.argv[1])
report={}
def measure(p):
 solids=cq.importers.importStep(str(p)).solids().vals()
 data=[]
 for s in solids:
  b=s.BoundingBox();data.append([s.Volume(),b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax])
 return sorted(data,key=lambda v:tuple(round(x,4) for x in v)),all(s.isValid() for s in solids)
for kind in ['Envelope','Frame']:
 baseline,_=measure(old/f'MG20_Tail_RevA_{kind}.step')
 for suffix in ['', '_FLAT']:
  p=here/f'MG20_Tail_RevA_{kind}{suffix}.step'
  rows,valid=measure(p)
  assert len(rows)==len(baseline)
  a=np.array(rows); b=np.array(baseline)
  ii,jj=linear_sum_assignment(np.linalg.norm(a[:,None,1:]-b[None,:,1:],axis=2))
  bounds_error=float(np.max(np.abs(a[ii,1:]-b[jj,1:])))
  vol_error=float(np.max(np.abs(a[ii,0]-b[jj,0])/b[jj,0]))
  assert bounds_error<.001,(p,bounds_error)
  assert vol_error<.001,(p,vol_error)
  text=p.read_text()
  counts={t:len(re.findall(r'\b'+t+r'\s*\(',text)) for t in ['PCURVE','SURFACE_CURVE','MANIFOLD_SOLID_BREP','NEXT_ASSEMBLY_USAGE_OCCURRENCE']}
  assert counts['PCURVE']==counts['SURFACE_CURVE']==0
  if suffix:assert counts['NEXT_ASSEMBLY_USAGE_OCCURRENCE']==0
  assert valid
  report[p.name]={'solids':len(rows),'valid':valid,'max_bbox_difference_mm':bounds_error,'max_relative_volume_difference':vol_error,'entities':counts}
(here/'KOMPAS_EXPORT_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
