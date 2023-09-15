#include "cellular.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_cellular", &UpdateCellular);
}