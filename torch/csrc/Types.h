#ifndef THP_TYPES_INC
#define THP_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <string>

template <typename T> struct THPTypeInfo {};

namespace torch {

typedef struct THVoidStorage
{
  void *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  void *allocator;
  void *allocatorContext;
  THVoidStorage *view;
} THVoidStorage;

typedef struct THVoidTensor
{
   long *size;
   long *stride;
   int nDimension;
   THVoidStorage *storage;
   ptrdiff_t storageOffset;
   int refcount;
   char flag;
} THVoidTensor;

struct THPVoidTensor {
  PyObject_HEAD
  THVoidTensor *cdata;
  char device_type;
  char data_type;
};

void _THVoidTensor_assertContiguous(THVoidTensor *tensor, const std::string &prefix, const std::string& name);

#define THVoidTensor_assertContiguous(tensor, prefix)			\
  _THVoidTensor_assertContiguous(tensor, prefix, #tensor " tensor")

}  // namespace torch

#endif
