#include <Halide.h>
#include <cstdlib>
#include <fstream>
#include <vector>


using namespace Halide;

int main(int argc, char **argv) {

  int batch = atoi(argv[1]), height = atoi(argv[2]), width = atoi(argv[3]);

  Buffer<float> guide(batch, height, width, "guide");
  Buffer<float> coeff(batch, 16, 16, 8, 12, "coeff");
  Buffer<float> grad_output(batch, height, width, 12, "grad_output");

  Var x("x"), y("y"), c("c"), n("n");
  RDom r(0, 16, 0, 16, 0, 8);
  float sx = 16.0f / height;
  float sy = 16.0f / width;

  Func affine;
  Expr taux = max(1 - abs(x * sx - r.x), 0);
  Expr tauy = max(1 - abs(y * sy - r.y), 0);
  Expr tauz = max(1 - abs(8 * guide(n, x, y) - r.z), 0);
  affine(n, x, y, c) = 0.0f;
  affine(n, x, y, c) += taux * tauy * tauz * coeff(n, r.x, r.y, r.z, c);

  // Propagate the gradients to inputs
  auto d = propagate_adjoints(affine, grad_output);
  Func grad_guide = d(guide);
  Func grad_coeff = d(coeff);

  SimpleAutoscheduleOptions options;
  options.gpu = true;

  std::vector<Func> grad_guide_and_coeff = {grad_guide, grad_coeff};

  simple_autoschedule(grad_guide_and_coeff, {},
                      {{{0, batch - 1}, {0, height - 1}, {0, width - 1}},
                       {{0, batch - 1}, {0, 15}, {0, 15}, {0, 7}, {0, 11}}},
                      options);

  Target target(Target::OS::Windows, Target::Arch::X86, 64,
                {Target::CUDA, Target::CUDACapability61});
  std::ofstream grad_guide_stream("slice_layer_backward_grad_guide.cpp");
  Internal::CodeGen_PyTorch pytorch_codegen_grad_guide(
      grad_guide_stream, target,
      Internal::CodeGen_PyTorch::PyTorchImplementation,
      "slice_layer_backward_grad_guide.h");
  grad_guide.compile_to_static_library(
      "slice_layer_backward_grad_guide", {guide, coeff, grad_output},
      "slice_layer_backward_grad_guide", target);
  auto grad_guide_module = grad_guide.compile_to_module(
      {guide, coeff, grad_output}, "slice_layer_backward_grad_guide", target);
  pytorch_codegen_grad_guide.compile(grad_guide_module);

  std::ofstream grad_coeff_stream("slice_layer_backward_grad_coeff.cpp");
  Internal::CodeGen_PyTorch pytorch_codegen_grad_coeff(
      grad_coeff_stream, target,
      Internal::CodeGen_PyTorch::PyTorchImplementation,
      "slice_layer_backward_grad_coeff.h");
  grad_coeff.compile_to_static_library(
      "slice_layer_backward_grad_coeff", {guide, coeff, grad_output},
      "slice_layer_backward_grad_coeff", target);
  auto grad_coeff_module = grad_coeff.compile_to_module(
      {guide, coeff, grad_output}, "slice_layer_backward_grad_coeff", target);
  pytorch_codegen_grad_coeff.compile(grad_coeff_module);

  return 0;
}