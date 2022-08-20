User Managed IP (RTL Kernel)
============================

Simple example of user managed RTL Kernel.

**KEY CONCEPTS:** `User-Managed RTL Kernel <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Creating-User-Managed-RTL-Kernels>`__

**KEYWORDS:** `package_xo <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/package_xo-Command>`__, `ctrl_protocol <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/package_xo-Command>`__, `user_managed <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Creating-User-Managed-RTL-Kernels>`__, `xrt::ip <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Setting-Up-User-Managed-Kernels-and-Argument-Buffers>`__, `xrt::xclbin <https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Specifying-the-Device-ID-and-Loading-the-XCLBIN>`__, xrt::kernel::get_kernels, xrt::kernel::get_cus, xrt::kernel::get_args, xrt::arg::get_offset, xrt::ip::write_register, xrt::ip::read_register

.. raw:: html

 <details>

.. raw:: html

 <summary> 

 <b>EXCLUDED PLATFORMS:</b>

.. raw:: html

 </summary>
|
..

 - All NoDMA Platforms, i.e u50 nodma etc

.. raw:: html

 </details>

.. raw:: html

DESIGN FILES
------------

Application code is located in the src directory. Accelerator binary files will be compiled to the xclbin directory. The xclbin directory is required by the Makefile and its contents will be filled during compilation. A listing of all the files in this example is shown below

::

   src/hdl/krnl_vadd_rtl.v
   src/hdl/krnl_vadd_rtl_adder.sv
   src/hdl/krnl_vadd_rtl_axi_read_master.sv
   src/hdl/krnl_vadd_rtl_axi_write_master.sv
   src/hdl/krnl_vadd_rtl_control_s_axi.v
   src/hdl/krnl_vadd_rtl_counter.sv
   src/hdl/krnl_vadd_rtl_int.sv
   src/host.cpp
   
COMMAND LINE ARGUMENTS
----------------------

Once the environment has been configured, the application can be executed by

::

   ./rtl_user_managed -x <vadd XCLBIN>

DETAILS
-------

This example demonstrates how a user can create a User-Managed RTL IP. The RTL IP here does simple vector addition where two vectors are transferred from host to kernel, added and the result is written back to the host and verified. The IP's control protocol is mentioned as user_managed by adding ``-ctrl_protocol user_managed`` to the package command as below: 

::

   package_xo -ctrl_protocol user_managed -xo_path ${xoname} -kernel_name krnl_vadd_rtl -ip_directory ./packaged_kernel_${suffix}

The IP is first created, the CU information fetched and the memory index determined as below:  

.. code:: cpp

   std::vector<xrt::xclbin::ip> cu;
   auto ip = xrt::ip(device, uuid, "krnl_vadd_rtl");
   auto xclbin = xrt::xclbin(binaryFile);
   std::cout << "Fetch compute Units" << std::endl;
   for (auto& kernel : xclbin.get_kernels()) {
       if (kernel.get_name() == "krnl_vadd_rtl") {
           cu = kernel.get_cus();
       }
   }
   
   if (cu.empty()) throw std::runtime_error("IP krnl_vadd_rtl not found in the provided xclbin");
   
   std::cout << "Determine memory index\n";
   for (auto& mem : xclbin.get_mems()) {
       if (mem.get_used()) {
           mem_used = mem;
           break;
       }
   }

All the IP settings are achieved using the ``write_register`` and ``read_register`` calls as below:

.. code:: cpp

   std::cout << "INFO: Setting IP Data" << std::endl;
   
   auto args = cu[0].get_args();
   
   std::cout << "Setting the 1st Register \"a\" (Input Address)" << std::endl;
   ip.write_register(args[0].get_offset(), buf_addr[0]);
   ip.write_register(args[0].get_offset() + 4, buf_addr[0] >> 32);
   
   std::cout << "Setting the 2nd Register \"b\" (Input Address)" << std::endl;
   ip.write_register(args[1].get_offset(), buf_addr[1]);
   ip.write_register(args[1].get_offset() + 4, buf_addr[1] >> 32);
   
   std::cout << "Setting the 3rd Register \"c\" (Output Address)" << std::endl;
   ip.write_register(args[2].get_offset(), buf_addr[2]);
   ip.write_register(args[2].get_offset() + 4, buf_addr[2] >> 32);
   
   std::cout << "Setting the 4th Register \"length_r\"" << std::endl;
   ip.write_register(args[3].get_offset(), DATA_SIZE);
   
   uint32_t axi_ctrl = 0;
   
   std::cout << "INFO: IP Start" << std::endl;
   axi_ctrl = IP_START;
   ip.write_register(CSR_OFFSET, axi_ctrl);
   
   // Wait until the IP is DONE
   
   axi_ctrl = 0;
   while ((axi_ctrl & IP_IDLE) != IP_IDLE) {
       axi_ctrl = ip.read_register(CSR_OFFSET);
   }
   
   std::cout << "INFO: IP Done" << std::endl;

RTL kernels can be integrated to Vitis using ``RTL Kernel Wizard``.
These kernels have the same software interface model as OpenCL and C/C++
kernels. That is, they are seen by the host application as functions
with a void return value, scalar arguments, and pointer arguments.

The RTL Kernel Wizard automates some of the steps that need to be taken
to ensure that the RTL IP is packaged into a kernel that can be
integrated into a system in Vitis environment.

For more comprehensive documentation, `click here <http://xilinx.github.io/Vitis_Accel_Examples>`__.