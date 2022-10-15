#ifndef __FPGA_H__
#define __FPGA_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	FPGA_OK = 0,         /**< Operation completed successfully */
	FPGA_INVALID_PARAM,  /**< Invalid parameter supplied */
	FPGA_BUSY,           /**< Resource is busy */
	FPGA_EXCEPTION,      /**< An exception occurred */
	FPGA_NOT_FOUND,      /**< A required resource was not found */
	FPGA_NO_MEMORY,      /**< Not enough memory to complete operation */
	FPGA_NOT_SUPPORTED,  /**< Requested operation is not supported */
	FPGA_NO_DRIVER,      /**< Driver is not loaded */
	FPGA_NO_DAEMON,      /**< FPGA Daemon (fpgad) is not running */
	FPGA_NO_ACCESS,      /**< Insufficient privileges or permissions */
	FPGA_RECONF_ERROR    /**< Error while reconfiguring FPGA */
} fpga_result;

typedef enum { 
	FPGA_DEVICE = 0,
	FPGA_ACCELERATOR
} fpga_objtype;

typedef void *fpga_handle;

typedef void *fpga_token;

typedef void *fpga_properties;

typedef uint8_t fpga_guid[16];

#ifdef __cplusplus
}
#endif

#endif // __FPGA_H__
