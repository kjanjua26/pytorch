#ifndef THCP_CUDA_NCCL_INC
#define THCP_CUDA_NCCL_INC
#include <Python.h>

PyObject* THCPModule_nccl_reduce(PyObject *self, PyObject *args);
PyObject* THCPModule_nccl_destroy(PyObject *args);


#endif
