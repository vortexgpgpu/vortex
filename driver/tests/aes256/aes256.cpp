#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <openssl/evp.h>
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

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n nblocks] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h?")) != -1) {
    switch (c) {
    case 'n':
      nblocks = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
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

void cleanup() {
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             const char *expected_dec,
             const char *expected_enc,
             uint32_t buf_size) { 
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  // download newly-decrypted buffer
  std::cout << "download decrypted buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.outdec_ptr, buf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  int errors = 0;
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    if (memcmp(buf_ptr, expected_dec, buf_size)) {
      std::cout << "decrypted data does not match" << std::endl;
      ++errors;
    }
  }

  // download newly-encrypted buffer
  std::cout << "download encrypted buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.outenc_ptr, buf_size, 0));

  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    if (memcmp(buf_ptr, expected_enc, buf_size)) {
      std::cout << "encrypted data does not match" << std::endl;
      ++errors;
    }
  }

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;  
  }

  return 0;
}

// Adapted from an example in POSIX.1-2001
static inline uint8_t myrand(void) {
  static uint32_t next = 1;
  next = next * 1103515245 + 12345;
  return (uint32_t)(next / 65536) & 0xff;
}

static int openssl_aes256(const char *expected_dec, const char *key,
                           char *expected_enc, uint32_t buf_size) {
    EVP_CIPHER_CTX *ctx;
    if (!(ctx = EVP_CIPHER_CTX_new())) {
        return 1;
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    if (!EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, (const unsigned char *)key, NULL)) {
        return 1;
    }

    int outl;
    if (!EVP_EncryptUpdate(ctx, (unsigned char *)expected_enc, &outl, (const unsigned char *)expected_dec, buf_size)) {
        return 1;
    }
    if (outl != buf_size) {
        return 1;
    }

    EVP_CIPHER_CTX_free(ctx);
    return 0;
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

  uint32_t num_tasks = max_cores * max_warps * max_threads;
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

  RT_CHECK(vx_alloc_dev_mem(device, KEY_SIZE, &value));
  kernel_arg.key_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.indec_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.inenc_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.outdec_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.outenc_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.nblocks = nblocks;

  std::cout << "key_ptr=0x" << std::hex << kernel_arg.key_ptr << std::dec << std::endl;
  std::cout << "indec_ptr=0x" << std::hex << kernel_arg.indec_ptr << std::dec << std::endl;
  std::cout << "inenc_ptr=0x" << std::hex << kernel_arg.inenc_ptr << std::dec << std::endl;
  std::cout << "outdec_ptr=0x" << std::hex << kernel_arg.outdec_ptr << std::dec << std::endl;
  std::cout << "outenc_ptr=0x" << std::hex << kernel_arg.outenc_ptr << std::dec << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(KEY_SIZE, std::max<uint32_t>(buf_size, sizeof(kernel_arg_t)));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  char *aes_key = (char *)malloc(KEY_SIZE);
  RT_CHECK(!aes_key);
  char *expected_dec = (char *)malloc(buf_size);
  RT_CHECK(!expected_dec);
  char *expected_enc = (char *)malloc(buf_size);
  RT_CHECK(!expected_enc);

  for (uint32_t i = 0; i < KEY_SIZE; i++) {
    aes_key[i] = myrand();
  }
  for (uint32_t i = 0; i < buf_size; i++) {
    expected_dec[i] = myrand();
  }
  //memset(aes_key, 0xff, KEY_SIZE);
  //memset(expected_dec, 0xff, buf_size);
  RT_CHECK(openssl_aes256(expected_dec, aes_key, expected_enc, buf_size));

  // upload key buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, aes_key, KEY_SIZE);
  }
  std::cout << "upload key buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.key_ptr, KEY_SIZE, 0));

  // upload decrypted input buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, expected_dec, buf_size);
  }
  std::cout << "upload decrypted input buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.indec_ptr, buf_size, 0));

  // upload encrypted input buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memcpy(buf_ptr, expected_enc, buf_size);
  }
  std::cout << "upload encrypted input buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.inenc_ptr, buf_size, 0));

  // clear decrypted output buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memset(buf_ptr, 0, buf_size);
  }
  std::cout << "clear decrypted output buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.outdec_ptr, buf_size, 0));  

  // clear encrypted output buffer
  {
    auto buf_ptr = (char*)vx_host_ptr(buffer);
    memset(buf_ptr, 0, buf_size);
  }
  std::cout << "clear encrypted output buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.outenc_ptr, buf_size, 0));  

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, expected_dec, expected_enc, buf_size));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();
  free(aes_key);
  free(expected_dec);
  free(expected_enc);

  std::cout << "PASSED!" << std::endl;

  return 0;
}
