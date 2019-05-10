// #include <torch/extension.h>
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

#define Accessor(size) \
  torch::PackedTensorAccessor<scalar_t, size, torch::RestrictPtrTraits, size_t>

#define accessor(size) \
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
  int n = blockIdx.z;
  int xo = blockIdx.x;
  int yo = blockIdx.y;

  int xi = threadIdx.x;
  int yi = threadIdx.y;

  float sx = static_cast<float>(guide.size(1)) / coeff.size(1);
  float sy = static_cast<float>(guide.size(2)) / coeff.size(2);

  int x = xo * 8 + xi;
  if (x >= output.size(1)) return;

  int y = yo * 8 + yi;
  if (y >= output.size(2)) return;

  scalar_t local_output[12];
  __shared__ scalar_t shared_coeff[8][8][8][12];

  for (int c = 0; c < 12; ++c) {
    local_output[c] = 0;
  }

  for (int io = 0; io < 2; ++io) {    // A.I = 16
    for (int jo = 0; jo < 2; ++jo) {  // A.J = 16
      int sii = threadIdx.x;
      int sji = threadIdx.y;

      // 8 * 12 mod 32 = 3，不存在内存体冲突
      for (int sk = 0; sk < 8; ++sk) {
        for (int sc = 0; sc < 12; ++sc) {
          shared_coeff[sii][sji][sk][sc] =
              coeff[n][io * 8 + sii][jo * 8 + sji][sk][sc];
        }
      }

      __syncthreads();

      for (int ii = 0; ii < 2; ++ii) {
        int i = io * 8 + ii;
        for (int ji = 0; ji < 2; ++ji) {
          int j = jo * 8 + ji;
          // 8 * 12 mod 32 = 3，不存在内存体冲突
          for (int k = 0; k < 8; ++k) {     // A.K = 8
            for (int c = 0; c < 12; ++c) {  // O.C = 12
              local_output[c] += tau(sx * x - i) * tau(sy * y - j) *
                                 tau(8 * guide[n][x][y] - k) *
                                 shared_coeff[i][j][k][c];
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
  const auto weight = guide.size(2);
  const auto channel = coeff.size(4);
  auto output = torch::zeros({batch, height, weight, channel});
  const dim3 blocks(guide.size(1), guide.size(2), guide.size(0));
  const dim3 threads(8, 8);
  AT_DISPATCH_FLOATING_TYPES(
      coeff.type(), "slice_cuda_forward", ([&] {
        slice_forward_kernel<scalar_t><<<blocks, threads>>>(
            coeff.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits,
                                  size_t>(),
            guide.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits,
                                  size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,
                                   size_t>());
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

  float sx = static_cast<float>(guide.size(1)) / grad_coeff.size(1);
  float sy = static_cast<float>(guide.size(2)) / grad_coeff.size(2);

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
//             O[x][y] += grad_slice[x][y][c] * tau(sx * x - i) * tau(sy * y -
//             j) *
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
// tau(sy * y - k) * d_tau(d * guide[n, x, y] - k) * d * coeff[n, i, j, k, c],
// reduce=[i, j, k, c])
template <typename scalar_t>
__global__ void slice_backward_kernel_grad_guide(const Accessor(5) coeff,
                                                 const Accessor(3) guide,
                                                 const Accessor(4) grad_slice,
                                                 Accessor(3) grad_guide) {
  int n = blockIdx.z;
  int xo = blockIdx.x;
  int yo = blockIdx.y;

  int xi = threadIdx.x;
  int yi = threadIdx.y;

  float sx = static_cast<float>(guide.size(1)) / coeff.size(1);
  float sy = static_cast<float>(guide.size(2)) / coeff.size(2);

  // grad_guide的height维度，以8为单位进行分割
  int x = xo * 8 + xi;
  if (x >= grad_slice.size(1)) return;

  // grad_guide的weight维度，以8为单位进行分割
  int y = yo * 8 + yi;
  if (y >= grad_slice.size(2)) return;

  scalar_t local_grad_guide = 0;
  // 共享内存大小必须在编译期间就确定大小
  __shared__ scalar_t shared_coeff[8][8][8][12];

  // 加载共享存储器内容

  // 对coeff的height维度，以8为单位进行分割，这里假设height维度大小为16
  for (int io = 0; io < 2; ++io) {    // A.I = 16
    for (int jo = 0; jo < 2; ++jo) {  // A.J = 16
      int sii = threadIdx.x;
      int sji = threadIdx.y;

      // 8 * 12 mod 32 = 3，不存在内存体冲突
      for (int sk = 0; sk < 8; ++sk) {
        for (int sc = 0; sc < 12; ++sc) {
          shared_coeff[sii][sji][sk][sc] =
              coeff[n][io * 8 + sii][jo * 8 + sji][sk][sc];
        }
      }

      __syncthreads();

      for (int ii = 0; ii < 8; ++ii) {
        int i = io * 8 + ii;
        for (int ji = 0; ji < 2; ++ji) {
          int j = jo * 8 + ji;
          // 8 * 12 mod 32 = 3，不存在内存体冲突
          for (int k = 0; k < 8; ++k) {     // A.K = 8
            for (int c = 0; c < 12; ++c) {  // O.C = 12
              local_grad_guide += tau(sx * x - i) * tau(sy * y - j) *
                                  d_tau(8 * guide[n][x][y] - k) * 8 *
                                  shared_coeff[i][j][k][c];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  grad_guide[n][x][y] = local_grad_guide;
}

std::vector<torch::Tensor> slice_cuda_backward(torch::Tensor grad_slice,
                                               torch::Tensor coeff,
                                               torch::Tensor guide) {
  auto grad_coeff = torch::zeros_like(grad_slice);
  auto grad_guide = torch::zeros_like(grad_slice);

  const dim3 blocks0(16, 16, coeff.size(0));
  const dim3 threads0(8, 12);
  AT_DISPATCH_FLOATING_TYPES(
      grad_slice.type(), "slice_cuda_backward_grad_coeff", ([&] {
        slice_backward_kernel_grad_coeff<scalar_t><<<blocks0, threads0>>>(
            guide.accessor(3), grad_slice.accessor(4), grad_coeff.accessor(5));
      }));

  const dim3 blocks1(guide.size(1), guide.size(2), guide.size(0));
  const dim3 threads1(8, 8);
  AT_DISPATCH_FLOATING_TYPES(
      grad_slice.type(), "slice_cuda_backward_grad_guide", ([&] {
        slice_backward_kernel_grad_guide<scalar_t><<<blocks1, threads1>>>(
            coeff.accessor(5), guide.accessor(3), grad_slice.accessor(4),
            grad_guide.accessor(3));
      }));

  return {grad_coeff, grad_guide};
}
