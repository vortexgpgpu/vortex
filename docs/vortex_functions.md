# Non-opencl kernels on Vortex

Vortex codes can be written either in OpenCL or C using Vortex-specific functions - [sample codes](https://github.com/vortexgpgpu/vortex/tree/master/tests/regression/)

## Essential Vortex functions

- `vx_dev_open()` - Open or start device connection 
- `vx_upload_kernel_file()` - Upload kernel programs
- `vx_mem_alloc()` - Allocate device side memory. Returns the start address of location which is passed in as a kernel argument later. 
- `vx_buf_alloc()` - Allocates "shared" memory. This is essentially the canvas/space that enables host <-> device transfer. 
- `vx_copy_to_dev()` - Copy host variables to device side using the address returned earlier by vx_mem_alloc()
- `vx_start()` and `vx_ready_wait()` - For kernel invocation
- `vx_copy_from_dev()` - Copy from device to host side

## General flow 

  ```C
  //1. Allocate and initialize host memory      
  //2. open device connection
  vx_device_h device = nullptr;
  vx_dev_open(&device);

  //3. Declare buffer sizes

  //4. upload program
  const char* kernel_file = "kernel.bin";
  vx_upload_kernel_file(device, kernel_file);

  //5. allocate device memory
  vx_mem_alloc(device, BufferSizeHere, &value);
  kernel_arg.variable1 = value;
  
  //6. allocate shared memory  
  vx_buf_alloc(device, SharedBufferSize, &Shared_buf);
  
  //7. upload kernel 1 arguments, 8. upload source buffers, 9. initialize dest buffer (same format)
	kernel_arg.variable1 = 0;
	{
	auto buf_ptr_upload = (int*)vx_host_ptr(Shared_buf);
      buf_ptr_upload = hostvariable;
  }
  vx_copy_to_dev(Shared_buf, kernel_arg.variable2, BufferSize, 0);  

  //10. Start device
	vx_start(device);

  //11. Wait for completion
  vx_ready_wait(device, MAX_TIMEOUT);
  
  //12. Download destination buffer for masks
  //13. Copy results to host arrays, print if necessary
    vx_copy_from_dev(Shared_buf, kernel_arg.variable2, BufferSize, 0);
    {
    auto buf_ptr1 = (int32_t*)vx_host_ptr(common_buf);
    for (uint32_t i = 0; i < no_of_nodes; ++i) {   
        hostvariable2[i] = buf_ptr1[i];
        std::cout << "Result index [" <<i<<"]is "<<hostvariable2[i]<<std::endl;
      }
    }

```
