#include "Halide.h"
#include "slice_layer_backward_grad_coeff.h"
#include "slice_layer_backward_grad_guide.h"
#include "slice_layer_forward_affine.h"

#include <chrono>
#include <map>
#include <numeric>
#include <vector>

using namespace std::chrono;
using namespace Halide;

class Benchmark {
 public:
  static std::map<std::string, std::vector<double>> runtime;

 private:
  static double mean(std::vector<double> data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  }

 public:
  Benchmark(std::function<void(void)> target, std::string name) {
    high_resolution_clock::time_point start = high_resolution_clock::now();
    target();
    high_resolution_clock::time_point end = high_resolution_clock::now();
    runtime[name].push_back(
        duration_cast<duration<double>>(end - start).count());
  }

  static void statistic() {
    for (auto record : runtime) {
      std::cout << record.first << " : " << mean(record.second) << std::endl;
    }
  }
};

std::map<std::string, std::vector<double>> Benchmark::runtime;

int main(int argc, char const *argv[]) {
  int batch = atoi(argv[1]), height = atoi(argv[2]), width = atoi(argv[3]);

  Buffer<float> guide(batch, height, width, "guide");
  Buffer<float> grad_guide(batch, height, width, "guide");
  Buffer<float> coeff(batch, 16, 16, 8, 12, "coeff");
  Buffer<float> grad_coeff(batch, 16, 16, 8, 12, "coeff");
  Buffer<float> affine(batch, height, width, 12, "affine");
  Buffer<float> grad_output(batch, height, width, 12, "grad_output");

  for (int _ = 0; _ < 10; _++) {
    Benchmark(
        [&] {
          slice_layer_forward_affine(guide.raw_buffer(), coeff.raw_buffer(),
                                     affine.raw_buffer());
        },
        "forward");
    Benchmark(
        [&] {
          slice_layer_backward_grad_coeff(
              guide.raw_buffer(), coeff.raw_buffer(), grad_output.raw_buffer(),
              grad_coeff.raw_buffer());
          slice_layer_backward_grad_guide(
              guide.raw_buffer(), coeff.raw_buffer(), grad_output.raw_buffer(),
              grad_guide.raw_buffer());
        },
        "backwward");
  }

  Benchmark::statistic();

  return 0;
}
