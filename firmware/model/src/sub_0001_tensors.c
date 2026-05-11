#include "sub_0001_tensors.h"

const TensorInfo sub_0001_tensors[] = {
  { "_split_1_command_stream", 0, 12792, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 2366240, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 772192, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 772192, "FAST_SCRATCH", 0x0 },
  { "input_70299_10706_70153", 4, 76800, "INPUT_TENSOR", 0x0 },
  { "logits_70207_10553", 5, 5, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0001_tensors_count = sizeof(sub_0001_tensors) / sizeof(sub_0001_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0001_address_input_70299_10706_70153 = 0x0;
const uint32_t sub_0001_address_logits_70207_10553 = 0x0;

