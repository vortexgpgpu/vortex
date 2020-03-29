#include <iostream>
#include <unistd.h>
#include <vortex.h>

const char* program_file = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: -f: program [-h: help]" << std::endl;
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
  int err;
  
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  vx_device_h device;
  err = vx_dev_open(&device);
  if (err != 0)
    return -1;

  // upload program
  err = vx_upload_kernel_file(device, program_file);
  if (err != 0) {
    vx_dev_close(device);
    return -1;  
  }

  // start device
  err = vx_start(device);
  if (err != 0) {
    vx_dev_close(device);
    return -1;  
  }

  // wait for completion
  err = vx_ready_wait(device, -1);
  if (err != 0) {
    vx_dev_close(device);
    return -1;  
  }

  // close device
  vx_dev_close(device);

  printf("done!\n");

  return 0;
}