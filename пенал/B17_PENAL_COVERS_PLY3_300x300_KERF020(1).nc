; Energy per length: +10 percent versus rib settings F480 / F232.8.
; M4 S1000 ONE PASS; $30=1000 $32=1; overrides 100 percent.
; MG20 B17 - FOUR COVERS - BIRCH PLYWOOD 3 mm
; Sheet 300x300 mm. Zero lower left. Maximum X235.200 Y198.400.
; Finished parts: 230x44.4 qty2, 150x44.4 qty2 mm.
; KERF 0.200 mm already applied. No extra compensation or sanding allowance.
; Face grain MUST run along machine Y, across cover width.
; Same power scale and feed override assumed; actual cut/kerf must be checked.
; $30=1000, $32=1. Outer-face bevels must be made separately.
G21
G90
G94
M5
; PASS 1/1
; COVER 1
G0 X5.000 Y5.000
M4 S1000
G1 F211.64
G1 X235.200 Y5.000
G1 X235.200 Y49.600
G1 X5.000 Y49.600
G1 X5.000 Y5.000
M5
; COVER 2
G0 X5.000 Y54.600
M4 S1000
G1 F211.64
G1 X235.200 Y54.600
G1 X235.200 Y99.200
G1 X5.000 Y99.200
G1 X5.000 Y54.600
M5
; COVER 3
G0 X5.000 Y104.200
M4 S1000
G1 F211.64
G1 X155.200 Y104.200
G1 X155.200 Y148.800
G1 X5.000 Y148.800
G1 X5.000 Y104.200
M5
; COVER 4
G0 X5.000 Y153.800
M4 S1000
G1 F211.64
G1 X155.200 Y153.800
G1 X155.200 Y198.400
G1 X5.000 Y198.400
G1 X5.000 Y153.800
M5
G0 X0 Y0
M5
