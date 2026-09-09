#include <CL/opencl.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

static void check(cl_int err) {
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL error " << err << std::endl;
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  std::string benchmark = "tl_cg";
  int n = 128, resources = 1, rounds = 4, percent = 100;
  int stride = 1, local = 32, software = 0;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) {
      return 2;
    }
    std::string arg = argv[i], value = argv[i + 1];
    if (arg == "--benchmark") {
      benchmark = value;
    } else if (arg == "--size") {
      n = std::stoi(value);
    } else if (arg == "--resources") {
      resources = std::stoi(value);
    } else if (arg == "--rounds") {
      rounds = std::stoi(value);
    } else if (arg == "--percent") {
      percent = std::stoi(value);
    } else if (arg == "--stride") {
      stride = std::stoi(value);
    } else if (arg == "--local") {
      local = std::stoi(value);
    } else if (arg == "--software") {
      software = std::stoi(value);
    } else {
      std::cerr << "Unknown argument " << arg << std::endl;
      return 2;
    }
  }
  const std::vector<std::string> names = {
    "tl_cg", "tl_fg", "am_cg", "am_fg", "lockht", "atm", "lclist", "cp_ds", "bh_st"
  };
  auto found = std::find(names.begin(), names.end(), benchmark);
  if (found == names.end() || n < 1 || resources < 1 || rounds < 1 ||
      local < 1 || n % local || stride < 1 || percent < 0 || percent > 100 ||
      ((benchmark == "atm" || benchmark == "cp_ds") && resources < 2)) {
    std::cerr << "Invalid workload parameters" << std::endl;
    return 2;
  }
  int bench = found - names.begin();
  if ((bench == 0 || bench == 2) && resources != 1) {
    return 2;
  }
  int side = static_cast<int>(std::sqrt(resources));
  if (bench == 7 && (side < 2 || side * side != resources)) {
    return 2;
  }
  int count = std::max({n * rounds + 4, (resources + 2) * stride, 2 * n + 4, 2 * resources + 4});
  std::vector<std::vector<int>> host(5, std::vector<int>(count, 0));
  auto& data = host[0];
  auto& links = host[1];
  auto& locks = host[2];
  auto& audit = host[4];
  if (bench == 4) {
    std::fill(data.begin(), data.end(), -1);
    std::fill(links.begin(), links.end(), -1);
  } else if (bench == 5) {
    std::fill(data.begin(), data.end(), n * rounds + 1);
  } else if (bench == 6) {
    for (int i = 0; i <= resources + 1; ++i) {
      data[i] = i;
      links[i] = i + 1;
    }
    links[resources + 1] = -1;
  } else if (bench == 7) {
    for (int i = 0; i < resources; ++i) {
      data[2 * i] = (i % side) * 1024 + (i % 3) * 128;
      data[2 * i + 1] = (i / side) * 1024 + (i % 5) * 64;
    }
  } else if (bench == 8) {
    locks[0] = 1;
    for (int i = n; i >= 1; --i) {
      int first = 8 * i - 6;
      data[i] = first > n ? 1 : 0;
      for (int child = first; child < first + 8 && child <= n; ++child) {
        data[i] += data[child];
      }
    }
  }
  auto expected = host;
  if (bench < 4) {
    for (int i = 0; i < n; ++i) {
      for (int r = 0; r < rounds; ++r) {
        int bucket = (i / 2) % resources;
        if (bench < 2) {
          ++expected[0][bucket];
        } else if (i < (n * percent) / 100) {
          expected[0][bucket] = std::max(expected[0][bucket], (i * 73 + r * 19) % 100003);
        }
      }
    }
  } else if (bench == 5) {
    for (int i = 0; i < n; ++i) {
      int a = (i / 2) % resources;
      int b = (a + 1 + i % (resources - 1)) % resources;
      expected[0][a] -= rounds;
      expected[0][b] += rounds;
    }
  } else if (bench == 6) {
    for (int i = 0; i < n; ++i) {
      expected[3][i] = 1;
      if (i < (n * percent) / 100) {
        ++expected[4][1 + (i * 17) % resources];
      }
    }
  } else if (bench == 8) {
    for (int i = 1; i <= n; ++i) {
      int parent = i == 1 ? 0 : (i - 2) / 8 + 1;
      expected[3][i] = expected[3][parent];
      if (i > 1) {
        for (int sibling = 8 * parent - 6; sibling < i; ++sibling) {
          expected[3][i] += data[sibling];
        }
      }
      expected[2][i] = 1;
    }
  }

  cl_platform_id platform;
  cl_device_id device;
  check(clGetPlatformIDs(1, &platform, nullptr));
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr));
  cl_int err;
  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  check(err);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  check(err);
  std::ifstream source_file("kernel.cl");
  std::string source((std::istreambuf_iterator<char>(source_file)), {});
  const char* source_ptr = source.c_str();
  cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, nullptr, &err);
  check(err);
  std::string options = "-DBENCH=" + std::to_string(bench) + " -DSOFTWARE=" + std::to_string(software);
  err = clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t length;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
    std::vector<char> log(length + 1, 0);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
    std::cerr << log.data() << std::endl;
    check(err);
  }
  cl_kernel kernel = clCreateKernel(program, "evaluate", &err);
  check(err);
  std::vector<cl_mem> buffers;
  for (int i = 0; i < 5; ++i) {
    buffers.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    count * sizeof(int), host[i].data(), &err));
    check(err);
    check(clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]));
  }
  const int params[] = {n, resources, rounds, percent, stride};
  for (int i = 0; i < 5; ++i) {
    check(clSetKernelArg(kernel, 5 + i, sizeof(int), &params[i]));
  }
  size_t global_size = n, local_size = local;
  std::cout << "WORKLOAD: benchmark=" << benchmark << " size=" << n
            << " resources=" << resources << " rounds=" << rounds
            << " percent=" << percent << " stride=" << stride
            << " local=" << local << " software=" << software << std::endl;
  check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr));
  check(clFinish(queue));
  for (int i = 0; i < 5; ++i) {
    check(clEnqueueReadBuffer(queue, buffers[i], CL_TRUE, 0, count * sizeof(int),
                             host[i].data(), 0, nullptr, nullptr));
  }
  bool valid = true;
  if (bench == 4) {
    std::vector<int> seen(n * rounds, 0);
    for (int b = 0; b < resources; ++b) {
      int node = data[b];
      for (int steps = 0; node != -1; ++steps) {
        if (node < 0 || node >= n * rounds || steps >= n * rounds ||
            ((node % n) / 2) % resources != b || seen[node]++) {
          valid = false;
          break;
        }
        node = links[node];
      }
    }
    valid &= std::all_of(seen.begin(), seen.end(), [](int x) { return x == 1; });
    valid &= locks == expected[2];
  } else if (bench == 7) {
    valid &= audit[0] == n * rounds;
    int horizontal = side * (side - 1), edges = 2 * horizontal;
    std::vector<int> seen(edges, 0);
    for (int i = 1; i <= n * rounds && valid; ++i) {
      int edge = audit[i];
      if (edge < 0 || edge >= edges) {
        valid = false;
        break;
      }
      ++seen[edge];
      int e = edge % horizontal;
      int a = edge < horizontal ? (e / (side - 1)) * side + e % (side - 1) : e;
      int b = a + (edge < horizontal ? 1 : side);
      float dx = expected[0][2 * b] - expected[0][2 * a];
      float dy = expected[0][2 * b + 1] - expected[0][2 * a + 1];
      float distance = std::sqrt(dx * dx + dy * dy);
      float scale = distance > 0 ? 0.5f * (distance - 1024.0f) / distance : 0;
      int cx = static_cast<int>(dx * scale), cy = static_cast<int>(dy * scale);
      expected[0][2 * a] += cx;
      expected[0][2 * a + 1] += cy;
      expected[0][2 * b] -= cx;
      expected[0][2 * b + 1] -= cy;
    }
    for (int a = 0; a < edges; ++a) {
      valid &= seen[a] == rounds * (n / edges + (a < n % edges));
    }
    valid &= data == expected[0] && locks == expected[2];
  } else {
    valid = host == expected;
  }
  for (auto buffer : buffers) {
    check(clReleaseMemObject(buffer));
  }
  check(clReleaseKernel(kernel));
  check(clReleaseProgram(program));
  check(clReleaseCommandQueue(queue));
  check(clReleaseContext(context));
  std::cout << (valid ? "PASSED!" : "FAILED!") << std::endl;
  return valid ? 0 : 1;
}
