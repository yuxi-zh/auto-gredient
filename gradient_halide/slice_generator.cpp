#include "Halide.h"

namespace {

using namespace Halide;

class SliceLayer : public Halide::Generator<SliceLayer> {
 public:
  Input<Buffer<float>> guide{"guide", 4, 1024, 1024, 4};
  Input<Buffer<float>> grid{"grid", 4, 16, 16, 8, 12};
  Input<Buffer<float>> adjoints{"adjoints", 4, 1024, 1024, 4};
  Output<Buffer<float>> grad_guide{"grad_guide", 4, 1024, 1024, 4};
  Output<Buffer<float>> grad_grid{"grad_grid", 4, 16, 16, 8, 12};

  void generate() {
    /* THE ALGORITHM */

    Var x("x"), y("y"), c("c"), n("n");
    Rom r(0, 16, 0, 16, 0, 8);
    float sx = 16.0f / 1024.0f;
    float sy = 16.0f / 1024.0f;

    Func affine;
    Expr taux = max(1 - abs(x * sx - r.x), 0);
    Expr tauy = max(1 - abs(y * sy - r.y), 0);
    Expr tuaz = max(1 - abs(8 * guide(n, x, y, c) - r.z), 0);
    affine(n, x, y, c) = 0;
    affine(n, x, y, c) += taux * tauy * tauz * grid(n, r.x, r.y, r.z, c);

    // Propagate the gradients to inputs
    auto d = propagate_adjoints(output, adjoints);
    grad_guide = d(guide);
    grad_grid = d(grid);

    SimpleAutoscheduleOptions options;
    options.gpu = true;

    simple_autoschedule(d_guide, {}, {{0, 3}, {0, 1023}, {0, 1023}, {0, 3}},
                        options);

    simple_autoschedule(d_grid, {}, {{0, 3}, {0, 15}, {0, 15}, {0, 7}, {0, 11}},
                        options);
  }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(SliceLayer, slice_layer)
