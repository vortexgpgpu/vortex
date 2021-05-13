#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include "common.h"
#include "aes256.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
uint32_t nblocks = 0;
aes_op_type_t aes_op_type = AES_OP_ECB_ENC;

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n nblocks] [-t op_type] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c, op_type_arg;
  while ((c = getopt(argc, argv, "n:k:t:h?")) != -1) {
    switch (c) {
    case 'n':
      nblocks = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 't':
      op_type_arg = atoi(optarg);
      if (op_type_arg < 0 || op_type_arg >= (int)AES_OP_COUNT) {
        show_usage();
        std::cout << "invalid arg to -t" << std::endl;
        exit(-1);
      }
      aes_op_type = (aes_op_type_t)op_type_arg;
      break;
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
}

static void cleanup() {
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
    vx_dev_close(device);
  }
}

static int openssl_aes256(int enc, int ctr, const char *iv, const char *input,
                          const char *key, char *output, uint32_t buf_size) {
  EVP_CIPHER_CTX *ctx;
  if (!(ctx = EVP_CIPHER_CTX_new())) {
    ERR_print_errors_fp(stderr);
    return 1;
  }
  if (!EVP_CipherInit_ex(ctx, iv? (ctr? EVP_aes_256_ctr() : EVP_aes_256_cbc())
                                  : EVP_aes_256_ecb(),
                         NULL, (const unsigned char *)key,
                         (const unsigned char *)iv, enc)) {
    ERR_print_errors_fp(stderr);
    return 1;
  }
  if (!ctr) {
    // Doesn't make sense to set this for CTR since it's effectively a
    // streaming cipher
    EVP_CIPHER_CTX_set_padding(ctx, 0);
  }

  int outl_update = 0;
  if (!EVP_CipherUpdate(ctx, (unsigned char *)output, &outl_update,
                         (const unsigned char *)input, buf_size)) {
    ERR_print_errors_fp(stderr);
    return 1;
  }

  int outl_final = 0;
  if (!EVP_CipherFinal_ex(ctx, (unsigned char *)output + outl_update, &outl_final)) {
    ERR_print_errors_fp(stderr);
    return 1;
  }

  if (outl_update + outl_final != buf_size) {
    std::cerr << "Wrong amount of data " << (enc? "encrypted" : "decrypted")
              << " by openssl: " << outl_update << " + " << outl_final
              << " != " << buf_size << std::endl;
    return 1;
  }

  EVP_CIPHER_CTX_free(ctx);
  return 0;
}

static int openssl_aes256_op(aes_op_type_t op_type, char *iv, const char *input, const char *key,
                             char *output, uint32_t buf_size, int *no_ref_impl) {
  *no_ref_impl = 0;

  switch (op_type) {
    case AES_OP_ECB_ENC: return openssl_aes256(1, 0, NULL, input, key, output, buf_size);
    case AES_OP_ECB_DEC: return openssl_aes256(0, 0, NULL, input, key, output, buf_size);
    case AES_OP_CBC_ENC: return openssl_aes256(1, 0, iv, input, key, output, buf_size);
    case AES_OP_CBC_DEC: return openssl_aes256(0, 0, iv, input, key, output, buf_size);
    case AES_OP_CTR_ENC: return openssl_aes256(1, 1, iv, input, key, output, buf_size);
    case AES_OP_CTR_DEC: return openssl_aes256(0, 1, iv, input, key, output, buf_size);

    case AES_OP_KEY_ENC:
    case AES_OP_KEY_DEC:
        *no_ref_impl = 1;
        std::cout << "skipping generating expected value for key expansion as "
                     "we have no reference implementation" << std::endl;
        return 0;

    default:
        std::cout << "unsupported aes op " << aes_op_type << std::endl;
        return 1;
  }
}

int run_test(const kernel_arg_t& kernel_arg, const char *input,
             uint32_t buf_size) {
  int no_ref_impl;
  char *expected_output = (char *)malloc(buf_size);
  RT_CHECK(!expected_output);
  RT_CHECK(openssl_aes256_op(aes_op_type, (char *)kernel_arg.iv, input,
                             (char *)kernel_arg.key, expected_output,
                             buf_size, &no_ref_impl));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  if (!no_ref_impl) {
      // download newly-decrypted buffer
      std::cout << "download output buffer" << std::endl;
      RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.out_ptr, buf_size, 0));

      // verify result
      std::cout << "verify result" << std::endl;
      int errors = 0;
      {
        auto buf_ptr = (char*)vx_host_ptr(buffer);
        if (memcmp(buf_ptr, expected_output, buf_size)) {
          std::cout << "output data does not match expected" << std::endl;
          ++errors;
        }
      }

      if (errors != 0) {
        std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
      }
  }

  free(expected_output);
  return 0;
}

// Adapted from an example in POSIX.1-2001
static inline uint8_t myrand(void) {
  static uint32_t next = 1;
  next = next * 1103515245 + 12345;
  return (uint32_t)(next / 65536) & 0xff;
}

int main(int argc, char *argv[]) {
  size_t value;
  kernel_arg_t kernel_arg;

  // parse command arguments
  parse_args(argc, argv);

  if (!nblocks) {
    nblocks = 1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  nblocks = nblocks / max_cores;
  uint32_t num_tasks = max_cores * max_warps * max_threads;
  // uint32_t num_tasks = max_warps * max_threads;
  uint32_t num_points = nblocks * num_tasks;
  // Size of each of the four {input,output} {en,de}crypted buffers
  uint32_t buf_size = num_points * BLOCK_SIZE;

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "each buffer has size: " << buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.in_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.out_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.nblocks = nblocks;
  kernel_arg.aes_op_type = aes_op_type;

  std::cout << "key_ptr=0x" << std::hex << kernel_arg.key_ptr << std::dec << std::endl;
  std::cout << "in_ptr=0x" << std::hex << kernel_arg.in_ptr << std::dec << std::endl;
  std::cout << "out_ptr=0x" << std::hex << kernel_arg.out_ptr << std::dec << std::endl;

  // allocate shared memory
  std::cout << "allocate shared memory" << std::endl;
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));

  for (uint32_t i = 0; i < KEY_SIZE; i++) {
    kernel_arg.key[i] = myrand();
  }
  for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
    kernel_arg.iv[i] = myrand();
  }

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  char *input = (char *)malloc(buf_size);
  RT_CHECK(!input);
  for (uint32_t i = 0; i < buf_size; i++) {
    input[i] = myrand();
  }

  // upload input buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, input, buf_size);
  }
  std::cout << "upload input buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.in_ptr, buf_size, 0));

  // clear output buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memset(buf_ptr, 0, buf_size);
  }
  std::cout << "clear output buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.out_ptr, buf_size, 0));

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, input, buf_size));

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();
  free(input);

  std::cout << "PASSED!" << std::endl;

  return 0;
}
