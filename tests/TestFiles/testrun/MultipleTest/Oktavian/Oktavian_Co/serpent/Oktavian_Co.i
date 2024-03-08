% --- surface definitions --- %
surf 1 cylx 0.0 0.0 5.55 
surf 2 cylx 0.0 0.0 5.75 
surf 3 sph 0.0 0.0 0.0 10.0
surf 4 sph 0.0 0.0 0.0 10.2
surf 5 sph 0.0 0.0 0.0 19.75
surf 6 sph 0.0 0.0 0.0 19.95
surf 7 sph 0.0 0.0 0.0 100.0
surf 8 px 8.32
% --- cell definitions --- %
cell 1 0 void ( ( -3 -8 ) : ( 8 -1 -6 ) ) 
cell 2 0 2 ( ( 3 -4 -8 ) : ( 8 1 -2 -6 ) ) 
cell 3 0 1 ( ( 4 -5 -8 ) : ( 8 2 -5 ) ) 
cell 4 0 2 ( ( 5 -6 -8 ) : ( 8 2 5 -6 ) ) 
cell 5 0 void ( 6 -7 ) 
cell 6 0 void ( 7 ) 
% --- material definitions --- %
% M1
mat 1 -1.94 rgb 0 208 31
27059 -9.906499e-01 
30064 -3.000000e-05 
28058 -1.021150e-03 
28060 -3.933450e-04 
28061 -1.710000e-05 
28062 -5.451000e-05 
28064 -1.389000e-05 
14028 -3.689200e-04 
14029 -1.873200e-05 
14030 -1.234800e-05 
26054 -7.013999e-05 
26056 -1.101050e-03 
26057 -2.542800e-05 
26058 -3.384000e-06 
20040 -2.908200e-03 
20042 -1.941000e-05 
20043 -4.050000e-06 
20044 -6.269999e-05 
20046 -1.200000e-07 
20048 -5.610000e-06 
25055 -2.000000e-03 
16032 -7.601599e-04 
16033 -5.999999e-06 
16034 -3.368000e-05 
16036 -1.600000e-07 
29063 -6.916999e-05 
29065 -3.083000e-05 
82204 -2.800000e-07 
82206 -4.820000e-06 
82207 -4.420000e-06 
82208 -1.048000e-05 
6012 -2.970555e-04 
6013 -2.965728e-06 
% M2
mat 2 -7.824 rgb 0 0 255
24050 -8.038249e-03 
24052 -1.550100e-01 
24053 -1.757680e-02 
24054 -4.375249e-03 
26054 -4.114879e-02 
26056 -6.459479e-01 
26057 -1.491780e-02 
26058 -1.985280e-03 
28058 -7.556549e-02 
28060 -2.910749e-02 
28061 -1.265400e-03 
28062 -4.033739e-03 
28064 -1.027860e-03 