#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <string>
#include <cudnn.h>
#include "../Types.h"
#include <ATen/Tensor.h>

namespace torch { namespace cudnn {

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);
cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);

}}  // namespace torch::cudnn

#endif
