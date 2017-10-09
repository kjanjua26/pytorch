#include "nccl.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/cuda/THCP.h"

#include <nccl.h>
#include <sstream>
#include <unordered_map>

#ifdef ncclGroupStart
#define NCCL_VERSION 2
#endif

static inline void CHECK(ncclResult_t status)
{
  if (status != ncclSuccess) {
    std::stringstream err;
    err << "NCCL Error " << status << ": " << ncclGetErrorString(status);
    throw std::runtime_error(err.str());
  }
}

// TODO: make this thread_local + add mutexes
std::unordered_map<std::string, std::unordered_map<int, ncclComm_t> > _communicators;
std::vector<ncclComm_t *> _all_communicators;

PyObject * THCPModule_nccl_destroy(PyObject *args) {
  for(auto comm: _all_communicators) {
    ncclCommDestroy(*comm);
  }
  Py_RETURN_NONE;
}

static std::unordered_map<int, ncclComm_t> _get_communicator(const int *devs, int ndevices) {
  std::stringstream hash;
  for (int i = 0; i < ndevices; i++) {
    hash << devs[i] << ",";
  }
  if (_communicators.find(hash.str()) == _communicators.end()) {
    ncclComm_t *comms = (ncclComm_t*) malloc(sizeof(ncclComm_t) * ndevices);
    assert(comms);
    CHECK(ncclCommInitAll(comms, ndevices, devs));
    _all_communicators.push_back(comms);
    for (int i = 0; i < ndevices; i++) {
      _communicators[hash.str()][devs[i]] = comms[i];
    }
  }
  return _communicators[hash.str()];
}

static void _check_inputs(THPObjectPtr &inputs, THPObjectPtr &outputs, int size_multiplier) {
  if (!PySequence_Check(inputs.get()) || !PySequence_Check(outputs.get())) {
    throw std::runtime_error("inputs and outputs have to be sequences of Tensors");
  }

  // len(inputs) == len(outputs)
  Py_ssize_t len = PySequence_Fast_GET_SIZE(inputs.get());

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != PySequence_Fast_GET_SIZE(outputs.get())) {
    throw std::runtime_error("inputs and outputs sequences have to be of the same length");
  }

  std::unordered_set<int> devices;
  int64_t nelement = -1;

  for (Py_ssize_t i = 0; i < len; i++) {
    auto input = PySequence_Fast_GET_ITEM(inputs.get(), i);
    auto output = PySequence_Fast_GET_ITEM(outputs.get(), i);

    // input and output are is_cuda
    if (!(THPUtils_checkDenseCudaTensor(input) && THPUtils_checkDenseCudaTensor(input))) {
      throw std::runtime_error("input and output elements have to be dense CUDA Tensors");
    }

    // TODO: check that all inputs / outputs must be of same Tensor type

    torch::THVoidTensor *input_ = ((torch::THPVoidTensor*) input)->cdata;
    torch::THVoidTensor *output_ = ((torch::THPVoidTensor*) output)->cdata;

    // they are contiguous
    torch::THVoidTensor_assertContiguous((torch::THVoidTensor*)input_, "nccl");
    torch::THVoidTensor_assertContiguous((torch::THVoidTensor*)output_, "nccl");

    int input_device = ((torch::THCVoidStorage*) input_->storage)->device;
    int output_device = ((torch::THCVoidStorage*)output_->storage)->device;

    // inputs must be on unique devices
    if (devices.find(input_device) != devices.end()) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.insert(input_device);

    // inputs and outputs must be on same device respectively
    if (input_device != output_device) {
      throw std::runtime_error("input and output must be on the same device");
    }

    // all inputs must be same size
    int64_t input_nelement = torch::THVoidTensor_nElement((torch::THVoidTensor*) input_);
    if (i == 0) {
      nelement = input_nelement;
    } else if (input_nelement != nelement) {
      throw std::runtime_error("all inputs must have the same number of elements");
    }
  
    // outputs have to be of size * size_multiplier
    int64_t output_nelement = torch::THVoidTensor_nElement((torch::THVoidTensor*) output_);
    if (output_nelement != input_nelement * size_multiplier) {
      throw std::runtime_error("output must be of size input_size * size_multiplier");
    }
  }
}

static ncclDataType_t _get_data_type(PyObject *obj) {
  if (PyObject_IsInstance(obj, THCPFloatTensorClass)) {
    return ncclFloat;
  } else if (PyObject_IsInstance(obj, THCPHalfTensorClass)) {
    return ncclHalf;
  } else if (PyObject_IsInstance(obj, THCPLongTensorClass)) {
    return ncclInt64;
  } else if (PyObject_IsInstance(obj, THCPIntTensorClass)) {
    return ncclInt;
  } else if (PyObject_IsInstance(obj, THCPCharTensorClass)) {
    return ncclChar;
  } else if (PyObject_IsInstance(obj, THCPByteTensorClass)) {
    return ncclChar;
  } 
  throw std::runtime_error("Unknown input type given to _get_data_type");
}

PyObject * THCPModule_nccl_reduce(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  THPObjectPtr inputs_list, outputs_list, streams_list;
  PyObject *_inputs, *_outputs, *_streams;
  int root, op, original_device;
  Py_ssize_t len;
  std::vector<torch::THVoidTensor*> inputs, outputs;
  std::vector <THCStream *> streams;
  std::vector<int> devices;
  int64_t count;
  ncclDataType_t data_type;
  std::unordered_map<int, ncclComm_t> comm;
  std::mutex* mutex;

  if (!PyArg_ParseTuple(args, "OOOii", &_inputs, &_outputs, &_streams, &root, &op)) {
    THPUtils_invalidArguments(args, NULL, "nccl_reduce", 1,
			      "(sequence[Tensor] inputs, sequence[Tensor]"
			      " outputs, sequence[torch.cuda.Stream or None], int root, int op");
  }

  inputs_list = THPObjectPtr(PySequence_Fast(_inputs, NULL));
  outputs_list = THPObjectPtr(PySequence_Fast(_outputs, NULL));
  streams_list = THPObjectPtr(PySequence_Fast(_streams, NULL));

  _check_inputs(inputs_list, outputs_list, 1);

  len = PySequence_Fast_GET_SIZE(inputs_list.get());
  if (PySequence_Fast_GET_SIZE(streams_list.get()) != len) {
    throw std::runtime_error("number of streams is not equal to number of inputs");
  }

  for (Py_ssize_t i = 0; i < len; i++) {
    auto input = ((torch::THPVoidTensor*) PySequence_Fast_GET_ITEM(inputs_list.get(), i));
    auto output = ((torch::THPVoidTensor*) PySequence_Fast_GET_ITEM(outputs_list.get(), i));

    inputs.push_back(input->cdata);
    outputs.push_back(output->cdata);
    devices.push_back(((torch::THCVoidStorage*)input->cdata->storage)->device);

    PyObject *_stream = PySequence_Fast_GET_ITEM(streams_list.get(), i);
    if (PyObject_IsInstance(_stream, THCPStreamClass)) {
      streams.push_back( ((THCPStream *)_stream)->cdata);
    } else if (_stream == Py_None) {
      streams.push_back(NULL);
    } else {
      std::runtime_error("Unknown data type found in stream list. Need THCStream or None");
    }
  }

  data_type = _get_data_type(PySequence_Fast_GET_ITEM(inputs_list.get(), 0));
  // TODO: release GIL at this line
  comm = _get_communicator(&devices[0], len);
  THCudaCheck(cudaGetDevice(&original_device));
  count = torch::THVoidTensor_nElement((torch::THVoidTensor*) inputs[0]);
  mutex = THCCachingAllocator_getCudaFreeMutex();
  mutex->lock();
#if NCCL_VERSION == 2
  ncclGroupStart();
#endif
  for (Py_ssize_t i = 0; i < len; i++) {
    int device = devices[i];
    if (device != original_device) {
      THCudaCheck(cudaSetDevice(device));
    }
    auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
    ncclResult_t result = ncclReduce(THVoidTensor_data(inputs[i]), THVoidTensor_data(outputs[i]),
		     count, data_type, (ncclRedOp_t) op, root, comm[i], stream);
    CHECK(result);
  }
  THCudaCheck(cudaSetDevice(original_device));
#if NCCL_VERSION == 2
  ncclGroupEnd();
#endif
  mutex->unlock();

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
