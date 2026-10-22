; O0230 REVC FIT COUPON - same tongue path as revised ribs
; Stock 40x40 mm; grain along X; zero lower left
; F232.8 S1000 M4; GRBL 1.1 $30=1000 $32=1
; One pass; overrides 100%; no scaling or extra compensation
; Expected tongue 9.7 x 9.8 based on user's previous measurements
G21
G90
G94
M5
G0 X5.000 Y5.000
M4 S1000
G1 F232.8
G1 X20.000 Y5.000
G1 X20.000 Y12.000
G1 X29.700 Y12.000
G1 X29.700 Y22.000
G1 X20.000 Y22.000
G1 X20.000 Y30.000
G1 X5.000 Y30.000
G1 X5.000 Y5.000
M5
G0 X0 Y0
M5
