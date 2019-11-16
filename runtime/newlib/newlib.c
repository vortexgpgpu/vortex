

#include "../io/vx_io.h"
#include "../fileio/fileio.h"
#include "../intrinsics/vx_intrinsics.h"

#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>

#define CLOSE  1
#define ISATTY 2
#define LSEEK  3
#define READ   4
#define WRITE  5
#define FSTAT  6

#define FILE_IO_WRITE 0x71000000
#define FILE_IO_READ  0x72000000


typedef void (*funct_t)(void);

funct_t trap_to_simulator = (funct_t) 0x70000000;

void upload(char ** ptr, char * src, int size)
{
	char * drain = *ptr;

	// *((int *) drain) = size;
	char * size_ptr = (char *) size;
	drain[0] = size_ptr[0];
	drain[1] = size_ptr[1];
	drain[2] = size_ptr[2];
	drain[3] = size_ptr[3];

	drain += 4;


	for (int i = 0; i < size; i++)
	{
		(*drain) = src[i];
		drain += 1;
	}

	*ptr = drain;
}

void download(char ** ptr, char * drain)
{
	char * src = *ptr;

	int size;

	// size = *((int *) src);
	char * size_ptr = (char *) size;
	size_ptr[0] = src[0];
	size_ptr[1] = src[1];
	size_ptr[2] = src[2];
	size_ptr[3] = src[3];


	src += 4;

	// vx_printf("newlib.c: Size of download: ", size);
	// vx_printf("newlib.c: Real size: ", sizeof(struct stat));

	for (int i = 0; i < size; i++)
	{
		drain[i] = (*src);
		src += 1;
	}

	*ptr = src;
}

void _close()
{
	//vx_print_str("Hello from _close\n");
}

int _fstat(int file, struct stat * st)
{
	// char * write_buffer = (char *) FILE_IO_WRITE;

	// int cmd_id = FSTAT;

	// upload((char **) &write_buffer, (char *) &cmd_id, sizeof(int));
	// upload((char **) &write_buffer, (char *) &file  , sizeof(int));

	// trap_to_simulator();

	// char * read_buffer = (char *) FILE_IO_READ;

	// unsigned value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_mode = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_dev = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_uid = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_gid = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_size = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_blksize = value;
	// download((char **) &read_buffer, (char *) &value);
	// st->st_blocks = value;

	//vx_print_str("Hello from fstat\n");
	st->st_mode = S_IFCHR;
	// st->st_mode = 33279;

	// vx_printf("st_mode: ", st->st_mode);
	// vx_printf("st_dev: ", st->st_dev);
	// vx_printf("st_ino: ", st->st_ino);
	// vx_printf("st_uid: ", st->st_uid);
	// vx_printf("st_gid: ", st->st_gid);
	// vx_printf("st_rdev: ", st->st_rdev);
	// vx_printf("st_size: ", st->st_size);
	// vx_printf("st_blksize: ", st->st_blksize);
	// vx_printf("st_blocks: ", st->st_blocks);


	return  0;
}

int _isatty (int file)
{
  //vx_print_str("Hello from _isatty\n");
  return 1;
}

void _lseek()
{

	//vx_print_str("Hello from _lseek\n");
}

void _read()
{

	//vx_print_str("Hello from _read\n");
}

int _write (int file, char *buf, int nbytes)
{

	// char * write_buffer = (char *) FILE_IO_WRITE;

	// int cmd_id = WRITE;

	// upload((char **) &write_buffer, (char *) &cmd_id, sizeof(int));
	// upload((char **) &write_buffer, (char *) &file  , sizeof(int));
	// upload((char **) &write_buffer, (char *)  buf  , nbytes);


	// trap_to_simulator();

	//vx_print_str("Hello from _write\n");

	int i;

	unsigned int volatile * const print_addr = (unsigned int *) 0x00010000;

	for (i = 0; i < nbytes; i++)
	{
		(*print_addr) = buf[i];
    }
        
	return nbytes;

}



static int heap_start = (int) 0x10000000;
static int head_end   = (int) 0x20000000;

void * _sbrk (int nbytes)
{
	//vx_print_str("Hello from _sbrk\n");
	//vx_printf("nbytes: ", nbytes);

	//if (nbytes < 0) //vx_print_str("nbytes less than zero\n");
	// printf("nBytes: %d\n", nbytes);

	if (nbytes < 0)
	{
		nbytes = nbytes * -1;
	}

	if (nbytes > 10240)
	{
		nbytes = 10240;
	}

  // if (((unsigned) head_end) > ((unsigned) (heap_ptr + nbytes)))
	if (true)
    {
		int base  = heap_start;
		heap_start  += nbytes;
		////vx_print_str("_sbrk returning: ");
		//vx_print_hex((unsigned) base);
		////vx_print_str("\n");
		return (void *) base;
    }
	else
    {
		errno = ENOMEM;
		return  (void *) -1;
    }
}       /* _sbrk () */


void _exit(int val)
{
	//vx_print_str("Hello from exit\n");
	vx_tmc(0);
}

void _open()
{
	//vx_print_str("ERROR: _open not yet implemented\n");
}

void _kill()
{
	//vx_print_str("ERROR: _kill not yet implemented\n");
}

unsigned _getpid()
{
	return vx_threadID();
}

void _unlink()
{
	//vx_print_str("ERROR: _unlink not yet implemented\n");
}

static int curr_time = 0;

int _gettimeofday()
{
	//vx_print_str("ERROR: _gettimeofday not yet implemented\n");
	return curr_time++;
}


void _link()
{
	//vx_print_str("ERROR: _link not yet implemented\n");
}




