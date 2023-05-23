/*
 * Copyright 2021 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "models/dscnn/dscnn.h"

#include <stdio.h>

#include "menu.h"
#include "models/dscnn/model_dscnn.h"
#include "tflite.h"

// The model classifies speech based on the greatest of 35 scores.
typedef struct {
  uint8_t backward;
  uint8_t bed;
  uint8_t bird;
  uint8_t cat;
  uint8_t dog;
  uint8_t down;
  uint8_t eight;
  uint8_t five;
  uint8_t follow;
  uint8_t forward;
  uint8_t four;
  uint8_t go;
  uint8_t happy;
  uint8_t house;
  uint8_t learn;
  uint8_t left;
  uint8_t marvin;
  uint8_t nine;
  uint8_t no;
  uint8_t off;
  uint8_t on;
  uint8_t one;
  uint8_t right;
  uint8_t seven;
  uint8_t sheila;
  uint8_t six;
  uint8_t stop;
  uint8_t three;
  uint8_t tree;
  uint8_t two;
  uint8_t up;
  uint8_t visual;
  uint8_t wow;
  uint8_t yes;
  uint8_t zero;
} SpeechCommandsResult;

// Initialize everything once
// deallocate tensors when done
static void dscnn_init(void) {
  tflite_load_model(model_dscnn, model_dscnn_len);
}

// Run classification, after input has been loaded
SpeechCommandsResult classify() {
  printf("Running dscnn\n");
  tflite_classify();

  // Process the inference results.
  int8_t* output = tflite_get_output();
  return (SpeechCommandsResult){
      static_cast<uint8_t>(output[0] + 128),
      static_cast<uint8_t>(output[1] + 128),
      static_cast<uint8_t>(output[2] + 128),
      static_cast<uint8_t>(output[3] + 128),
      static_cast<uint8_t>(output[4] + 128),
      static_cast<uint8_t>(output[5] + 128),
      static_cast<uint8_t>(output[6] + 128),
      static_cast<uint8_t>(output[7] + 128),
      static_cast<uint8_t>(output[8] + 128),
      static_cast<uint8_t>(output[9] + 128),
      static_cast<uint8_t>(output[10] + 128),
      static_cast<uint8_t>(output[11] + 128),
      static_cast<uint8_t>(output[12] + 128),
      static_cast<uint8_t>(output[13] + 128),
      static_cast<uint8_t>(output[14] + 128),
      static_cast<uint8_t>(output[15] + 128),
      static_cast<uint8_t>(output[16] + 128),
      static_cast<uint8_t>(output[17] + 128),
      static_cast<uint8_t>(output[18] + 128),
      static_cast<uint8_t>(output[19] + 128),
      static_cast<uint8_t>(output[20] + 128),
      static_cast<uint8_t>(output[21] + 128),
      static_cast<uint8_t>(output[22] + 128),
      static_cast<uint8_t>(output[23] + 128),
      static_cast<uint8_t>(output[24] + 128),
      static_cast<uint8_t>(output[25] + 128),
      static_cast<uint8_t>(output[26] + 128),
      static_cast<uint8_t>(output[27] + 128),
      static_cast<uint8_t>(output[28] + 128),
      static_cast<uint8_t>(output[29] + 128),
      static_cast<uint8_t>(output[30] + 128),
      static_cast<uint8_t>(output[31] + 128),
      static_cast<uint8_t>(output[32] + 128),
      static_cast<uint8_t>(output[33] + 128),
      static_cast<uint8_t>(output[34] + 128),
  };
}

static void do_classify_zeros() {
  tflite_set_input_zeros();
  SpeechCommandsResult res = classify();
  printf("RESULTS-- \n"); 
  printf("backward: %d \n", res.backward);
  printf("bed: %d \n", res.bed);
  printf("bird: %d \n", res.bird);
  printf("cat: %d \n", res.cat);
  printf("dog: %d \n", res.dog);
  printf("down: %d \n", res.down);
  printf("eight: %d \n", res.eight);
  printf("five: %d \n", res.five);
  printf("follow: %d \n", res.follow);
  printf("forward: %d \n", res.forward);
  printf("four: %d \n", res.four);
  printf("go: %d \n", res.go);
  printf("happy: %d \n", res.happy);
  printf("house: %d \n", res.house);
  printf("learn: %d \n", res.learn);
  printf("left: %d \n", res.left);
  printf("marvin: %d \n", res.marvin);
  printf("nine: %d \n", res.nine);
  printf("no: %d \n", res.no);
  printf("off: %d \n", res.off);
  printf("on: %d \n", res.on);
  printf("one: %d \n", res.one);
  printf("right: %d \n", res.right);
  printf("seven: %d \n", res.seven);
  printf("sheila: %d \n", res.sheila);
  printf("six: %d \n", res.six);
  printf("stop: %d \n", res.stop);
  printf("three: %d \n", res.three);
  printf("tree: %d \n",  res.tree);
  printf("two: %d \n", res.two);
  printf("up: %d \n", res.up);
  printf("visual: %d \n", res.visual);
  printf("wow: %d \n", res.wow);
  printf("yes: %d \n", res.yes);
  printf("zero: %d \n", res.zero);
}

static struct Menu MENU = {
    "Tests for DS-CNN model",
    "dscnn",
    {
        MENU_ITEM('1', "Run with zeros input", do_classify_zeros),
        MENU_END,
    },
};

// For integration into menu system
void dscnn_menu() {
  dscnn_init();
  menu_run(&MENU);
}
