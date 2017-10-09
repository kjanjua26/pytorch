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

int64_t THVoidTensor_nElement(THVoidTensor *tensor) {
  if (tensor->nDimension == 0) {
    return 0;
  }

  int64_t nElement = 1;
  for (int i = 0; i < tensor->nDimension; i++) {
    nElement = nElement * tensor->size[i];
  }
  return nElement;
}

void* THVoidTensor_data(THVoidTensor *tensor) {
  if (tensor->storage) {
    return (void*) ((ptrdiff_t)tensor->storage->data + tensor->storageOffset);
  }
  return NULL;
}

}
