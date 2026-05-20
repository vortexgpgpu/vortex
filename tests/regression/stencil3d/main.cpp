
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                          \
    do                                                           \
    {                                                            \
        int _ret = _expr;                                        \
        if (0 == _ret)                                           \
            break;                                               \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
        cleanup();                                               \
        exit(-1);                                                \
    } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator
{
};

template <>
class Comparator<int>
{
public:
    static const char *type_str()
    {
        return "integer";
    }
    static int generate()
    {
        return rand();
    }
    static bool compare(int a, int b, int index, int errors)
    {
        if (a != b)
        {
            if (errors < 100)
            {
                printf("*** error: [%d] expected=%d, actual=%d\n", index, a, b);
            }
            return false;
        }
        return true;
    }
};

template <>
class Comparator<float>
{
private:
    union Float_t
    {
        float f;
        int i;
    };

public:
    static const char *type_str()
    {
        return "float";
    }
    static float generate()
    {
        return static_cast<float>(rand()) / RAND_MAX;
    }
    static bool compare(float a, float b, int index, int errors)
    {
        union fi_t
        {
            float f;
            int32_t i;
        };
        fi_t fa, fb;
        fa.f = a;
        fb.f = b;
        auto d = std::abs(fa.i - fb.i);
        if (d > FLOAT_ULP)
        {
            if (errors < 100)
            {
                printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
            }
            return false;
        }
        return true;
    }
};

static void stencil_cpu(TYPE *out, const TYPE *in, uint32_t width, uint32_t height, uint32_t depth)
{
    // We'll need to handle boundary conditions. Let's assume we use boundary replication.
    for (uint32_t z = 0; z < depth; z++)
    {
        for (uint32_t y = 0; y < height; y++)
        {
            for (uint32_t x = 0; x < width; x++)
            {
                TYPE sum = 0;
                int count = 0;

                // Iterate over the neighborhood
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            // Compute the neighbor's index
                            int nx = (int)x + dx;
                            int ny = (int)y + dy;
                            int nz = (int)z + dz;

                            // Check bounds and replicate the boundary values
                            if (nx < 0)
                            {nx = 0;}
                            else if (nx >= (int)width)
                            {nx = width - 1;}

                            if (ny < 0)
                            {ny = 0;}
                            else if (ny >= (int)height)
                            {ny = height - 1;}

                            if (nz < 0)
                            {nz = 0;}
                            else if (nz >= (int)depth)
                            {nz = depth - 1;}

                            // Sum up the values
                            sum += in[nz * width * height + ny * width + nx];
                            count++;
                        }
                    }
                }

                // Write the averaged value to the output array
                out[z * width * height + y * width + x] = sum / count;
            }
        }
    }
}

const char *kernel_file = "kernel.vxbin";
uint32_t size = 64;
uint32_t block_size = 2;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage()
{
    std::cout << "Vortex Test." << std::endl;
    std::cout << "Usage: [-k: kernel] [-n matrix_size] [-b:block_size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "n:t:k:h")) != -1)
    {
        switch (c)
        {
        case 'n':
            size = atoi(optarg);
            break;
        case 'b':
            block_size = atoi(optarg);
            break;
        case 'k':
            kernel_file = optarg;
            break;
        case 'h':
            show_usage();
            exit(0);
            break;
        default:
            show_usage();
            exit(-1);
        }
    }
}

void cleanup()
{
    if (device)
    {
        if (A_buffer) vx_buffer_release(A_buffer);
        if (B_buffer) vx_buffer_release(B_buffer);
        if (kernel)  vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue)   vx_queue_release(queue);
        vx_device_release(device);
    }
}

int main(int argc, char *argv[])
{
    // parse command arguments
    parse_args(argc, argv);

    if ((size / block_size) * block_size != size)
    {
        printf("Error: matrix size %d must be a multiple of block size %d\n", size, block_size);
        return -1;
    }

    std::srand(50);

    // open device connection
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint32_t size_cubed = size * size * size;
    uint32_t buf_size = size_cubed * sizeof(TYPE);

    std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
    std::cout << "matrix size: " << size << "x" << size << "x" << size << std::endl;
    std::cout << "block size: " << block_size << "x" << block_size << "x" << block_size << std::endl;

    kernel_arg.size = size;

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &A_buffer));
    RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &B_buffer));
    RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));

    std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
    std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;

    // allocate host buffers
    std::cout << "allocate host buffers" << std::endl;
    std::vector<TYPE> h_A(size_cubed);
    std::vector<TYPE> h_B(size_cubed);

    // generate source data
    for (uint32_t i = 0; i < size_cubed; ++i)
    {
        h_A[i] = Comparator<TYPE>::generate();
    }

    // upload source buffer0
    std::cout << "upload source buffer0" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), buf_size, 0, nullptr, nullptr));

    // load kernel module
    std::cout << "load kernel module" << std::endl;
    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    // start device
    std::cout << "start device" << std::endl;
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    {
        uint32_t grid_dim[3]  = {size / block_size, size / block_size, size / block_size};
        uint32_t block_dim[3] = {block_size, block_size, block_size};
        vx_launch_info_t li = {};
        li.struct_size  = sizeof(li);
        li.kernel       = kernel;
        li.args_host    = &kernel_arg;
        li.args_size    = sizeof(kernel_arg);
        li.ndim         = 3;
        li.grid_dim[0]  = grid_dim[0];
        li.grid_dim[1]  = grid_dim[1];
        li.grid_dim[2]  = grid_dim[2];
        li.block_dim[0] = block_dim[0];
        li.block_dim[1] = block_dim[1];
        li.block_dim[2] = block_dim[2];
        RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_enqueue_read(queue, h_B.data(), B_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // verify result
    std::cout << "verify result" << std::endl;
    int errors = 0;
    {
        std::vector<TYPE> h_ref(size_cubed);
        stencil_cpu(h_ref.data(), h_A.data(), size, size, size);

        for (uint32_t i = 0; i < h_ref.size(); ++i)
        {
            if (!Comparator<TYPE>::compare(h_B[i], h_ref[i], i, errors))
            {
                ++errors;
            }
        }
    }

    // cleanup
    std::cout << "cleanup" << std::endl;
    cleanup();

    if (errors != 0)
    {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return errors;
    }

    std::cout << "PASSED!" << std::endl;

    return 0;
}
