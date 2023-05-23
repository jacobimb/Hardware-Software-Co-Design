/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "perf.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

// Assumes 100 MHz
uint32_t ticks_per_second() { return 100000000 / 1024; }

// 1 "tick" = 1024 clock cycles
uint32_t GetCurrentTimeTicks() { return perf_get_mcycle64() >> 10; }

}  // namespace tflite
