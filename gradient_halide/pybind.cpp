#include <torch/extension.h>
#include <chrono>
#include <vector>

// CUDA forward declarations

int slice_layer_backward_grad_coeff_th_(at::Tensor &_guide, at::Tensor &_coeff,
                                        at::Tensor &_grad_output,
                                        at::Tensor &_coeff_d__);

int slice_layer_backward_grad_guide_th_(at::Tensor &_guide, at::Tensor &_coeff,
                                        at::Tensor &_grad_output,
                                        at::Tensor &_guide_d__);

int slice_layer_forward_affine_th_(at::Tensor &_guide, at::Tensor &_coeff,
                                   at::Tensor &_affine);

// C++ interface

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor slice_forward(torch::Tensor coeff, torch::Tensor guide) {
  // std::cout << "slice_forward is called!" << std::endl;
  CHECK_INPUT(coeff);
  CHECK_INPUT(guide);
  auto batch = guide.size(0);
  auto height = guide.size(1);
  auto width = guide.size(2);

  auto affine = torch::zeros({batch, height, width, 12},
                             torch::TensorOptions().device(torch::kCUDA));

  return affine;
}

std::vector<torch::Tensor> slice_backward(torch::Tensor grad_sliced,
                                          torch::Tensor coeff,
                                          torch::Tensor guide) {
  // std::cout << "slice_backward is called!" << std::endl;

  CHECK_INPUT(grad_sliced);
  CHECK_INPUT(coeff);
  CHECK_INPUT(guide);

  auto grad_guide = torch::zeros_like(guide);
  auto grad_coeff = torch::zeros_like(coeff);

  slice_layer_backward_grad_coeff_th_(guide, coeff, grad_sliced, grad_coeff);
  slice_layer_backward_grad_guide_th_(guide, coeff, grad_sliced, grad_guide);

  return {grad_guide, grad_coeff};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &slice_forward, "Slice forward (CUDA)");
  m.def("backward", &slice_backward, "Slice backward (CUDA)");
}