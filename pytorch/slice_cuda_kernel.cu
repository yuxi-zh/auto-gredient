//#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

// 不倾向于对归约的索引变量进行线程块或者线程绑定
// 因为绑定后涉及到线程块间或者线程块内的结果同步

// 前向计算过程中，只有x，y和c三个索引变量适合绑定线程块和线程
// 线程块的大小影响线程块占用的硬件资源，包括寄存器、共享内存
// 倾向于从小线程块开始尝试，线程块的大小必须为32的倍数

// for (int x = 0; x < O.X; ++x) {
//   for (int y = 0; y < O.Y; ++y) {
//     for (int c = 0; c < O.C; ++c) {  // O.C = 12
//       A[x][y][c] = 0;
//       for (int i = 0; i < A.I; ++i) {      // A.I = 16
//         for (int j = 0; j < A.J; ++j) {    // A.J = 16
//           for (int k = 0; k < A.K; ++k) {  // A.K = 8
//             A[x][y][c] += tau(sx * x - i) * tau(sy * y - j) *
//                           tau(d * G[x][y] - k) * A[i][j][k][c];
//           }
//         }
//       }
//     }
//   }
// }

// split(x, 8)
// split(y, 8)
// reorder(c)
// bind(xo, blockIdx.x)
// bind(yo, blockIdx.y)
// bind(xi, threadIdx.x)
// bind(xo, threadIdx.y)
// 16 * 16 * 8 * 12 * 4 / 1024 = 96
// 128 / 64 = 2
// 2 * 96 > 96, 实际只能用一半的CUDA核心，需要对i，j，k进行分块后再做内存提升
// 8 * 16 * 8 * 12 * 4 / 1024 = 48
// 2 * 48 = 96，正好用满共享内存和CUDA核心

#define Accessor(size)                                                         \
  torch::PackedTensorAccessor<scalar_t, size, torch::RestrictPtrTraits, size_t>

#define accessor(size)                                                         \
  packed_accessor<scalar_t, size, torch::RestrictPtrTraits, size_t>()

template <typename scalar_t>
__device__ __forceinline__ scalar_t tau(scalar_t x) {
  return max(1 - abs(x), static_cast<scalar_t>(0));
}

// 还可以进一步考虑循环展开和向量化加载与存储
template <typename scalar_t>
__global__ void slice_forward_kernel(const Accessor(5) coeff,
                                     const Accessor(3) guide,
                                     Accessor(4) output) {

  float sx = 16.0f / guide.size(1);
  float sy = 16.0f / guide.size(2);
  // for (int n = 0; n < output.size(0); ++n) {
  int n = blockIdx.z;
  // for (int xo = 0; xo < output.size(1) / 8; ++xo) {
  int xo = blockIdx.x;
  // for (int xi = 0; xi < 8; ++xi) {
  int xi = threadIdx.x;
  int x = xo * 8 + xi;
  // for (int yo = 0; yo < output.size(2) / 8; ++yo) {
  int yo = blockIdx.y;
  // for (int yi = 0; yi < 8; ++yi) {
  int yi = threadIdx.y;
  int y = yo * 8 + yi;

  scalar_t local_output[12];
  for (int c = 0; c < 12; ++c) {
    local_output[c] = 0;
  }

  __shared__ scalar_t shared_coeff[8][8][8][12];

  for (int io = 0; io < coeff.size(1) / 8; ++io) {   // coeff.size(I = 16
    for (int jo = 0; jo < coeff.size(2) / 8; ++jo) { // coeff.size(J = 16

      int si = threadIdx.x;
      int sj = threadIdx.y;
      int i = io * 8 + si;
      int j = jo * 8 + sj;
      for (int k = 0; k < 8; ++k) {
        for (int c = 0; c < 12; ++c) {
          shared_coeff[si][sj][k][c] = coeff[n][i][j][k][c];
        }
      }

      __syncthreads();

      for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ++ji) {
          for (int k = 0; k < coeff.size(3); ++k) { // A.K = 8
            for (int c = 0; c < 12; ++c) {          // O.C = 12
              local_output[c] += tau(sx * x - i) * tau(sy * y - j) *
                                 tau(8 * guide[n][x][y] - k) *
                                 shared_coeff[ii][ji][k][c];
            }
          }
        }
      }

      __syncthreads();
    }
  }

  for (int c = 0; c < 12; ++c) {
    output[n][x][y][c] = local_output[c];
  }
}

// coeff [batch, height, width, depth, channel]
// guide [batch, height, width]
torch::Tensor slice_cuda_forward(torch::Tensor coeff, torch::Tensor guide) {
  const auto batch = guide.size(0);
  const auto height = guide.size(1);
  const auto width = guide.size(2);
  const auto channel = coeff.size(4);
  // std::cout << "[" << batch << "," << height << "," << width << "," <<
  // channel
  //           << "]" << std::endl;
  auto output = torch::zeros({batch, height, width, channel},
                             torch::TensorOptions().device(torch::kCUDA));
  const dim3 blocks(guide.size(1) / 8, guide.size(2) / 8, guide.size(0));
  const dim3 threads(8, 8);
  // std::cout << "blocks=[" << blocks.x << "," << blocks.y << "," << blocks.z
  //           << "]" << std::endl;
  // std::cout << "threads=[" << threads.x << "," << threads.y << "," <<
  // threads.z
  //           << "]" << std::endl;

  AT_DISPATCH_FLOATING_TYPES(coeff.type(), "slice_cuda_forward", ([&] {
                               slice_forward_kernel<scalar_t>
                                   <<<blocks, threads>>>(coeff.accessor(5),
                                                         guide.accessor(3),
                                                         output.accessor(4));
                             }));
  return output;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sign(scalar_t x) {
  return x > 0 ? -1 : 1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tau(scalar_t x) {
  return abs(x) > 1 ? 0 : -sign(x);
}

// for (int i = 0; i < A.I; ++i) {        // A.I = 16
//   for (int j = 0; j < A.J; ++j) {      // A.J = 16
//     for (int k = 0; k < A.K; ++j) {    // A.K = 8
//       for (int c = 0; c < A.C; ++c) {  // A.C = 12
//         grad_coeff[i][j][k][c] = 0;
//         for (int x = 0; x < G.X; ++x) {
//           for (int y = 0; y < G.Y; ++y) {
//             grad_coeff[i][j][k][c] +=
//                 tau(sx * x - i) * tau(sy * y - j) * tau(8 * guide[x, y] -
//                 k);
//           }
//         }
//       }
//     }
//   }
// }

// 共256*n个线程块，每个线程块96个线程
// 还可以进一步内存提升、循环展开和向量化加载与存储
template <typename scalar_t>
__global__ void slice_backward_kernel_grad_coeff(const Accessor(3) guide,
                                                 const Accessor(4) grad_slice,
                                                 Accessor(5) grad_coeff) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  int n = blockIdx.z;

  int k = threadIdx.x;
  int c = threadIdx.y;

  float sx = grad_coeff.size(1) / static_cast<float>(guide.size(1));
  float sy = grad_coeff.size(2) / static_cast<float>(guide.size(2));

  scalar_t local_grad_coeff = 0;
  for (int x = 0; x < guide.size(1); ++x) {
    for (int y = 0; y < guide.size(2); ++y) {
      local_grad_coeff += grad_slice[n][x][y][c] * tau(sx * x - i) *
                          tau(sy * y - j) * tau(8 * guide[n][x][y] - k);
    }
  }
  grad_coeff[n][i][j][k][c] = local_grad_coeff;
}

//
// for (int x = 0; x < O.X; ++x) {
//   for (int y = 0; y < O.Y; ++y) {
//     O[x][y] = 0;
//     for (int c = 0; c < O.C; ++c) {        // O.C = 12
//       for (int i = 0; i < A.I; ++i) {      // A.I = 16
//         for (int j = 0; j < A.J; ++j) {    // A.J = 16
//           for (int k = 0; k < A.K; ++k) {  // A.K = 8
//             O[x][y] += grad_slice[x][y][c] * tau(sx * x - i) * tau(sy * y
//             - j) *
//                        d_tau(d * G[x][y] - k) * d * A[i][j][k][c];
//           }
//         }
//       }
//     }
//   }
// }

// 和前向计算十分相似，仅在最内层的计算上略有区别
// coeff [batch, height, width, depth, channel] = [n, 16, 16, 8, 12]
// guide [batch, height, width] = [n, x, y]
// grad_slice [batch, height, width, channel] = [n, x, y, 12]
// grad_guide [batch, height, width, depth, channel] = [n, 16, 16, 8, 12]
// grad_guide[n, x, y] = sum(grad_slice[n, x, y, c] * tau(sx * x - i) *
// tau(sy * y - k) * d_tau(d * guide[n, x, y] - k) * d * coeff[n, i, j, k,
// c], reduce=[i, j, k, c])
template <typename scalar_t>
__global__ void slice_backward_kernel_grad_guide(const Accessor(5) coeff,
                                                 const Accessor(3) guide,
                                                 const Accessor(4) grad_slice,
                                                 Accessor(3) grad_guide) {

  float sx = 16.0f / guide.size(1);
  float sy = 16.0f / guide.size(2);
  // for (int n = 0; n < output.size(0); ++n) {
  int n = blockIdx.z;
  // for (int xo = 0; xo < output.size(1) / 8; ++xo) {
  int xo = blockIdx.x;
  // for (int xi = 0; xi < 8; ++xi) {
  int xi = threadIdx.x;
  int x = xo * 8 + xi;
  // for (int yo = 0; yo < output.size(2) / 8; ++yo) {
  int yo = blockIdx.y;
  // for (int yi = 0; yi < 8; ++yi) {
  int yi = threadIdx.y;
  int y = yo * 8 + yi;

  scalar_t local_grad_guide;
  local_grad_guide = 0;

  __shared__ scalar_t shared_coeff[8][8][8][12];

  for (int io = 0; io < coeff.size(1) / 8; ++io) {   // coeff.size(I = 16
    for (int jo = 0; jo < coeff.size(2) / 8; ++jo) { // coeff.size(J = 16

      int si = threadIdx.x;
      int sj = threadIdx.y;
      int i = io * 8 + si;
      int j = jo * 8 + sj;
      for (int k = 0; k < 8; ++k) {
        for (int c = 0; c < 12; ++c) {
          shared_coeff[si][sj][k][c] = coeff[n][i][j][k][c];
        }
      }

      __syncthreads();

      for (int ii = 0; ii < 8; ++ii) {
        for (int ji = 0; ji < 8; ++ji) {
          for (int k = 0; k < coeff.size(3); ++k) { // A.K = 8
            for (int c = 0; c < 12; ++c) {          // O.C = 12
              local_grad_guide += tau(sx * x - i) * tau(sy * y - j) *
                                  d_tau(8 * guide[n][x][y] - k) * 8 *
                                  shared_coeff[ii][ji][k][c];
            }
          }
        }
      }

      __syncthreads();
    }
  }

  grad_guide[n][x][y] = local_grad_guide;
}

std::vector<torch::Tensor> slice_cuda_backward(torch::Tensor grad_slice,
                                               torch::Tensor coeff,
                                               torch::Tensor guide) {
  auto grad_coeff = torch::zeros_like(coeff);
  auto grad_guide = torch::zeros_like(guide);

  const dim3 blocks0(16, 16, coeff.size(0));
  const dim3 threads0(8, 12);
  AT_DISPATCH_FLOATING_TYPES(
      grad_slice.type(), "slice_cuda_backward_grad_coeff", ([&] {
        slice_backward_kernel_grad_coeff<scalar_t><<<blocks0, threads0>>>(
            guide.accessor(3), grad_slice.accessor(4), grad_coeff.accessor(5));
      }));

  const dim3 blocks1(guide.size(1) / 8, guide.size(2) / 8, guide.size(0));
  const dim3 threads1(8, 8);
  AT_DISPATCH_FLOATING_TYPES(
      grad_slice.type(), "slice_cuda_backward_grad_guide", ([&] {
        slice_backward_kernel_grad_guide<scalar_t><<<blocks1, threads1>>>(
            coeff.accessor(5), guide.accessor(3), grad_slice.accessor(4),
            grad_guide.accessor(3));
      }));

  return {grad_coeff, grad_guide};
}
