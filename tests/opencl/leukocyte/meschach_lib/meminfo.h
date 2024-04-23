
/**************************************************************************
**
** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
**
**			     Meschach Library
** 
** This Meschach Library is provided "as is" without any express 
** or implied warranty of any kind with respect to this software. 
** In particular the authors shall not be liable for any direct, 
** indirect, special, incidental or consequential damages arising 
** in any way from use of the software.
** 
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.  
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
***************************************************************************/


/* meminfo.h  26/08/93 */
/* changed  11/12/93 */


#ifndef MEM_INFOH
#define MEM_INFOH



/* for hash table in mem_stat.c */
/* Note: the hash size should be a prime, or at very least odd */
#define MEM_HASHSIZE         509
#define MEM_HASHSIZE_FILE    "meminfo.h"


/* default: memory information is off */
/* set it to 1 if you want it all the time */
#define MEM_SWITCH_ON_DEF	0


/* available standard types */
#define TYPE_NULL              (-1)
#define TYPE_MAT    	        0
#define TYPE_BAND               1
#define TYPE_PERM		2
#define TYPE_VEC		3
#define TYPE_IVEC		4

#ifdef SPARSE
#define TYPE_ITER		5
#define TYPE_SPROW              6
#define TYPE_SPMAT		7
#endif

#ifdef COMPLEX
#ifdef SPARSE
#define TYPE_ZVEC		8
#define TYPE_ZMAT		9
#else
#define TYPE_ZVEC		5
#define TYPE_ZMAT		6
#endif
#endif

/* structure for memory information */
typedef struct {
   long bytes;       /* # of allocated bytes for each type (summary) */
   int  numvar;      /* # of allocated variables for each type */
} MEM_ARRAY;



#ifdef ANSI_C

int  mem_info_is_on(void);
int mem_info_on(int sw);

long mem_info_bytes(int type,int list);
int mem_info_numvar(int type,int list);
void mem_info_file(FILE * fp,int list);

void mem_bytes_list(int type,int old_size,int new_size,
		       int list);
void mem_numvar_list(int type, int num, int list);

#ifndef THREADSAFE
int mem_stat_reg_list(void **var,int type,int list,char *fname,int line);
int mem_stat_mark(int mark);
int mem_stat_free_list(int mark,int list);
int mem_stat_show_mark(void);
void mem_stat_dump(FILE *fp,int list);
int mem_attach_list(int list,int ntypes,char *type_names[],
	int (*free_funcs[])(), MEM_ARRAY info_sum[]);
int mem_free_vars(int list);
int mem_is_list_attached(int list);
void mem_dump_list(FILE *fp,int list);
int mem_stat_reg_vars(int list,int type,char *fname,int line,...);
#endif /* THREADSAFE */
#else
int mem_info_is_on();
int mem_info_on();

long mem_info_bytes();
int mem_info_numvar();
void mem_info_file();

void mem_bytes_list();
void mem_numvar_list();

#ifndef THREADSAFE
int mem_stat_reg_list();
int mem_stat_mark();
int mem_stat_free_list();
int mem_stat_show_mark();
void mem_stat_dump();
int mem_attach_list();
int mem_free_vars();
int mem_is_list_attached();
void mem_dump_list();
int mem_stat_reg_vars();
#endif /* THREADSAFE */

#endif 

/* macros */

#define mem_info()   mem_info_file(stdout,0)

#ifndef THREADSAFE
#define mem_stat_reg(var,type)  mem_stat_reg_list((void **)var,type,0,__FILE__,__LINE__)
#define MEM_STAT_REG(var,type)  mem_stat_reg_list((void **)&(var),type,0,__FILE__,__LINE__)
#define mem_stat_free(mark)   mem_stat_free_list(mark,0)
#else
#define mem_stat_reg(var,type)
#define MEM_STAT_REG(var,type)
#define mem_stat_free(mark)
#endif

#define mem_bytes(type,old_size,new_size)  \
  mem_bytes_list(type,old_size,new_size,0)

#define mem_numvar(type,num) mem_numvar_list(type,num,0)


/* internal type */

typedef struct {
   char **type_names;        /* array of names of types (strings) */
   int  (**free_funcs)();    /* array of functions for releasing types */
   unsigned ntypes;          /* max number of types */
   MEM_ARRAY *info_sum;      /* local array for keeping track of memory */
} MEM_CONNECT;

/* max number of lists of types */
#define MEM_CONNECT_MAX_LISTS    5


#endif
