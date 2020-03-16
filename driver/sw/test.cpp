#include <vx_driver.h>
#include <iostream>
#include <unistd.h>

#include "utils.h"

#define CACHE_LINESIZE		64

const char* program_file = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-f: program] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "f:h?")) != -1) {
    switch (c) {
    case 'f': {
      program_file = optarg;
    } break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }

  if (nullptr == program_file) {
    show_usage();
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  auto device = vx_dev_open();

  // upload program
  if (0 != upload_program(device, program_file)) {
    vx_dev_close(device);
    return -1;  
  }

  // start device
  if (0 != vx_start(device)) {
    vx_dev_close(device);
    return -1;  
  }

  // wait for completion
  if (0 != vx_ready_wait(device, -1)) {
    vx_dev_close(device);
    return -1;  
  }

  // close device
  vx_dev_close(device);

  return 0;
}