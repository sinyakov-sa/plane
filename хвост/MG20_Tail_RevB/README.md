# MG20 Tail RevB — preliminary CAD geometry

Coordinates: X aft, Y right, Z up; units mm. Stabilizer centered on Y=0 at Z=0, vertical fin centered at Y=0.

## Included STEP files
- `MG20_Tail_RevB_4Solids.step`: a simple compound of 4 closed solids, no assembly hierarchy.
- `H_Stabilizer.step`, `H_Elevator.step`, `V_Fin.step`, `V_Rudder.step`: four standalone solids.
- `generate.py`: reproducible CadQuery script.

## Parameters
- Horizontal tail span 1000 mm; root chord 200; tip chord 120. Elevator chord 60 mm root / 36 mm tip (~30%).
- Vertical tail height 420 mm; root chord 300; tip chord 180. Rudder chord 90 mm root / 54 mm tip (~30%).
- Simplified constant thickness: fixed surfaces 16 mm; control surfaces 10 mm. **Not an aerodynamic airfoil.**
- Moment arm 800 mm is an assumed placement parameter, not reflected by a fuselage/wing assembly here.

## Validation
Exported with CadQuery and checked by re-import through OpenCascade STEPControl_Reader: assembled file has 4 solid bodies; each component file has 1 solid.

## Important remaining work
This is a *geometric envelope/prototype*, not a structural design or cut-ready model. Airfoil sections, spars/ribs, hinges/gaps, control horns, fittings to fuselage, servo loads, flutter, strength and CG margins require separate design and validation. The STEP files have not been tested in the KOMPAS application itself.

## KOMPAS diagnostic
First open `H_Stabilizer.step`. If its 1-solid file also opens blank, investigate KOMPAS import/version settings; otherwise open the 4-solid combined file. Do not confuse the GUI's blank display with proof that the STEP lacks entities.
