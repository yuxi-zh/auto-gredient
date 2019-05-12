```
['slice_layer.py', '4', '64', '64']
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunction                                    8184.780us      43141.818us               10      81847.800us     431418.181us
view                                                6.976us          2.139us              120        837.100us        256.702us
mul                                              1415.057us       8675.546us              140     198108.000us    1214576.488us
sub                                                55.007us         83.277us               30       1650.200us       2498.314us
abs                                                66.037us         83.230us               40       2641.500us       3329.214us
rsub                                               66.443us         75.098us               30       1993.300us       2252.952us
gt                                                 41.067us         22.397us               30       1232.000us        671.909us
_th_gt                                             15.403us          6.519us               30        462.100us        195.559us
where                                              91.145us        129.603us               40       3645.800us       5184.110us
_s_where                                           66.830us         75.254us               40       2673.200us       3010.141us
sum                                                60.037us      12380.224us               30       1801.100us     371406.723us
struct torch::autograd::GraphRoot                   4.900us          1.913us               10         49.000us         19.127us
SliceFunctionBackward                           14086.920us     117741.680us               10     140869.200us    1177416.801us
lt                                                 41.880us         69.888us               10        418.800us        698.883us
_th_lt                                             16.790us          8.527us               10        167.900us         85.274us
sign                                               38.060us         25.800us               10        380.600us        258.003us
neg                                                34.500us         71.646us               10        345.000us        716.461us
struct torch::autograd::AccumulateGrad             29.010us         56.660us               20        580.200us       1133.209us

------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunctionBasedCuda                            331.390us       3821.335us               10       3313.900us      38213.353us
struct torch::autograd::GraphRoot                   5.250us          2.238us               10         52.500us         22.379us
SliceFunctionBasedCudaBackward                    184.200us      25662.826us               10       1842.000us     256628.262us
struct torch::autograd::AccumulateGrad             35.015us         74.302us               20        700.300us       1486.042us

['slice_layer.py', '4', '128', '128']
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunctionBasedCuda                            389.470us       9304.549us               10       3894.700us      93045.491us
struct torch::autograd::GraphRoot                   5.900us          1.825us               10         59.000us         18.253us
SliceFunctionBasedCudaBackward                    224.180us     105977.460us               10       2241.800us    1059774.597us
struct torch::autograd::AccumulateGrad             30.085us         66.836us               20        601.700us       1336.716us

['slice_layer.py', '4', '256', '256']
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunctionBasedCuda                            467.430us      31163.695us               10       4674.300us     311636.948us
struct torch::autograd::GraphRoot                   5.990us          1.580us               10         59.900us         15.800us
SliceFunctionBasedCudaBackward                    225.910us     406108.389us               10       2259.100us    4061083.885us
struct torch::autograd::AccumulateGrad             30.675us         62.918us               20        613.500us       1258.362us

['slice_layer.py', '4', '512', '512']
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunctionBasedCuda                            783.400us     118222.400us               10       7834.000us    1182224.005us
struct torch::autograd::GraphRoot                   5.660us          1.289us               10         56.600us         12.894us
SliceFunctionBasedCudaBackward                    288.290us    1634473.640us               10       2882.900us   16344736.404us
struct torch::autograd::AccumulateGrad             32.200us         71.155us               20        644.000us       1423.096us

['slice_layer.py', '4', '1024', '1024']
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                                               CPU time        CUDA time            Calls        CPU total       CUDA total
------------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------
SliceFunctionBasedCuda                           1936.340us     470073.234us               10      19363.400us    4700732.344us
struct torch::autograd::GraphRoot                   5.140us          0.775us               10         51.400us          7.751us
SliceFunctionBasedCudaBackward                    402.510us    6575782.330us               10       4025.100us   65757823.303us
struct torch::autograd::AccumulateGrad             30.810us        163.818us               20        616.200us       3276.367us
```