

#include "../io/vx_io.h"
#include "../fileio/fileio.h"


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

void upload(char ** ptr, char * src, int size)
{
	char * drain = *ptr;

	*drain = size;
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
	size = *((int *) src);
	src += 4;

	for (int i = 0; i < size; i++)
	{
		drain[i] = (*src);
		src += 1;
	}

	*ptr = src;
}

void _close()
{
	vx_print_str("Hello from _close\n");
}

int _fstat(int file, struct stat * st)
{
	char * write_buffer = (char *) FILE_IO_WRITE;

	int cmd_id = FSTAT;

	upload((char **) &write_buffer, (char *) &cmd_id, sizeof(int));
	upload((char **) &write_buffer, (char *) &file  , sizeof(int));

	vx_fstat();

	char * read_buffer = (char *) FILE_IO_READ;

	download((char **) &read_buffer, (char *) st);
}

int _isatty (int file)
{
  vx_print_str("Hello from _isatty\n");
  return 1;
}

void _lseek()
{

	vx_print_str("Hello from _lseek\n");
}

void _read()
{

	vx_print_str("Hello from _read\n");
}

int _write (int file, char *buf, int nbytes)
{

	char * write_buffer = (char *) FILE_IO_WRITE;

	int cmd_id = WRITE;

	upload((char **) &write_buffer, (char *) &cmd_id, sizeof(int));
	upload((char **) &write_buffer, (char *) &file  , sizeof(int));

	upload((char **) &write_buffer, (char *)  buf  , nbytes);

	vx_write();


	// int i;

	// unsigned int volatile * const print_addr = (unsigned int *) 0x00010000;

	// for (i = 0; i < nbytes; i++)
	// {
	// 	(*print_addr) = buf[i];
 //    }
        
	// return nbytes;

}



static int heap_start = (int) 0x10000000;
static int head_end   = (int) 0x20000000;

void * _sbrk (int nbytes)
{
	//vx_print_str("Hello from _sbrk\n");
	//vx_printf("nbytes: ", nbytes);

	//if (nbytes < 0) vx_print_str("nbytes less than zero\n");
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
		//vx_print_str("_sbrk returning: ");
		//vx_print_hex((unsigned) base);
		//vx_print_str("\n");
		return (void *) base;
    }
	else
    {
		errno = ENOMEM;
		return  (void *) -1;
    }
}       /* _sbrk () */





