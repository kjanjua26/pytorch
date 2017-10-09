#include "Types.h"

#include <stdexcept>

namespace torch {

void _THVoidTensor_assertContiguous(THVoidTensor *tensor, const std::string& prefix, const std::string& name)
{
  static const std::string error_str = prefix + " requires contiguous ";
  // Contiguity check
  int64_t expectedStride = 1;
  for (int i = tensor->nDimension-1; i >= 0; --i) {
    if (tensor->size[i] != 1) {
      if (tensor->stride[i] != expectedStride)
        throw std::invalid_argument(error_str + name);
      expectedStride *= tensor->size[i];
    }
  }
}

}
