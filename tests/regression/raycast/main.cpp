#include "common.h"
#include "tracer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

const char *kernel_file = "kernel.vxbin";
const char *output_file = "output.ppm";
const char *model_file = "teapot.obj";
//const char *model_file = "sphere.obj";

uint32_t mesh_count = 1;

bool use_cpu = false;

//uint32_t dst_width = 160;
//uint32_t dst_height = 120;
uint32_t dst_width = 80;
uint32_t dst_height = 60;

float camera_fvf = 45;
float camera_zoom = 1.0f;

float3_t light_pos = float3_t(0, 10, -10);
float3_t light_color = float3_t(1, 1, 1.0);
float3_t ambient_color = float3_t(0.4f, 0.4f, 0.4f);

float3_t background_color = float3_t(0.4f, 0.35f, 0.25f);

uint32_t samples_per_pixel = 1;
uint32_t max_depth = 1;

static void show_usage() {
  std::cout << "Vortex Raycaster" << std::endl;
  std::cout << "Usage: [-k kernel] [-n meshes] [-w width] [-h height]" << std::endl;
  std::cout << "       [-m model] [-s samples] [-d depth] [-f vfov] [-z zoom] [-c]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int opt;
  while ((opt = getopt(argc, argv, "k:n:m:w:h:s:f:z:0d:o:c")) != -1) {
    switch (opt) {
      case 'k':
        kernel_file = optarg;
        break;
      case 'n':
        mesh_count = atoi(optarg);
        break;
      case 'm':
        model_file = optarg;
        break;
      case 'w':
        dst_width = atoi(optarg);
        break;
      case 'h':
        dst_height = atoi(optarg);
        break;
      case 'f':
        camera_fvf = atof(optarg);
        break;
      case 'z':
        camera_zoom = atof(optarg);
        break;
      case 's':
        samples_per_pixel = atoi(optarg);
        break;
      case 'd':
        max_depth = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'c':
        use_cpu = true;
        break;
      default:
        show_usage();
        exit(-1);
    }
  }

  // Handle any remaining non-option arguments if needed
  if (optind < argc) {
    std::cout << "Non-option arguments: ";
    while (optind < argc) {
      std::cout << argv[optind++] << " ";
    }
    std::cout << std::endl;
    show_usage();
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  auto start_time = std::chrono::high_resolution_clock::now();

  Tracer tracer(dst_width, dst_height, samples_per_pixel, max_depth, use_cpu);
  RT_CHECK(tracer.init(kernel_file, model_file, mesh_count));
  RT_CHECK(tracer.setup(camera_fvf, camera_zoom, light_pos, light_color, ambient_color, background_color));
  RT_CHECK(tracer.run(output_file));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
  std::cout << "Execution time: " << duration << " s" << std::endl;

  return 0;
}
