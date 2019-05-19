all: slice_generator

slice_generator: slice_generator.cpp
	g++ slice_generator.cpp -g -std=c++11 -I $(HL_ROOT)/include -L $(HL_ROOT)/bin -lHalide -lpthread -ldl -o slice_generator

slice_layer_test: slice_layer_test.cpp
	g++ slice_layer_test.cpp slice_layer_backward_grad_coeff.a slice_layer_backward_grad_guide.a slice_layer_forward_affine.a -g -std=c++11 -I $(HL_ROOT)/include -L $(HL_ROOT)/bin -lHalide -lpthread -ldl  -o slice_layer_test