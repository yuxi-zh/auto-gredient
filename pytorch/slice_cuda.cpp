#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor slice_cuda_forward(torch::Tensor coeff, torch::Tensor guide);

std::vector<torch::Tensor> slice_cuda_backward(
    torch::Tensor grad_sliced, torch::Tensor ceoff torch::Tensor guide);

// C++ interface

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> slice_forward(torch::Tensor coeff,
                                         torch::Tensor guide) {
  CHECK_INPUT(coeff);
  CHECK_INPUT(guide);

  return slice_cuda_forward(coeff, guide);
}

std::vector<torch::Tensor> slice_backward(torch::Tensor grad_sliced,
                                          torch::Tensor ceoff,
                                          torch::Tensor guide) {
  CHECK_INPUT(grad_sliced);
  CHECK_INPUT(ceoff);
  CHECK_INPUT(guide);

  return slice_cuda_backward(grad_sliced, ceoff, guide);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &slice_forward, "Slice forward (CUDA)");
  m.def("backward", &slice_backward, "Slice backward (CUDA)");
}