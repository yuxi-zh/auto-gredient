all: slice_generator

slice_generator: slice_generator.cpp
	g++ slice_generator.cpp -g -std=c++11 -I $(HL_ROOT)/include -L $(HL_ROOT)/bin -lHalide -lpthread -ldl -o slice_generator