
/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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


/* err.h  28/09/1993 */

/*  RCS id: $Id: err.h,v 1.2 1995/01/30 14:48:05 des Exp $  */


#ifndef ERRHEADER
#define ERRHEADER


#include        <setjmp.h>
#include        "machine.h"

/* Error recovery */

extern	jmp_buf	restart;


/* max. # of error lists */
#define ERR_LIST_MAX_LEN   10

/* main error functions */
#ifndef ANSI_C
extern	int ev_err();			/* main error handler */
extern	int set_err_flag();		/* for different ways of handling
                                                errors, returns old value */
extern  int count_errs();		/* to avoid "too many errors" */
extern  int err_list_attach();		/* for attaching a list of errors */
extern  int err_is_list_attached();	/* checking if a list is attached */
extern  int err_list_free();		/* freeing a list of errors */

#else  /* ANSI_C */

extern	int ev_err(const char *,int,int,const char *,int);  /* main error handler */
extern	int set_err_flag(int flag);         /* for different ways of handling
                                                errors, returns old value */
extern  int count_errs(int true_false);     /* to avoid "too many errors" */
extern  int err_list_attach(int list_num, int list_len,
	       char **err_ptr,int warn);  /* for attaching a list of errors */
extern  int err_is_list_attached(int list_num);  /* checking if a list 
						    is attached */
extern  int err_list_free(int list_num);   /* freeing a list of errors */

#endif


/* error(E_TYPE,"myfunc") raises error type E_TYPE for function my_func() */
#define	error(err_num,fn_name)	ev_err(__FILE__,err_num,__LINE__,fn_name,0)

/* warning(WARN_TYPE,"myfunc") raises warning type WARN_TYPE for 
   function my_func() */
#define warning(err_num,fn_name) ev_err(__FILE__,err_num,__LINE__,fn_name,1) 


/* error flags */
#define	EF_EXIT		0	/* exit on error */
#define	EF_ABORT	1	/* abort (dump core) on error */
#define	EF_JUMP		2	/* jump on error */
#define	EF_SILENT	3	/* jump, but don't print message */
#define	ERREXIT()	set_err_flag(EF_EXIT)
#define	ERRABORT()	set_err_flag(EF_ABORT)
/* don't print message */
#define	SILENTERR()	if ( ! setjmp(restart) ) set_err_flag(EF_SILENT)
/* return here on error */
#define	ON_ERROR()	if ( ! setjmp(restart) ) set_err_flag(EF_JUMP)


/* error types */
#define	E_UNKNOWN	0
#define	E_SIZES		1
#define	E_BOUNDS	2
#define	E_MEM		3
#define	E_SING		4
#define	E_POSDEF	5
#define	E_FORMAT	6
#define	E_INPUT		7
#define	E_NULL		8
#define	E_SQUARE	9
#define	E_RANGE		10
#define	E_INSITU2	11
#define	E_INSITU	12
#define	E_ITER		13
#define	E_CONV		14
#define	E_START		15
#define	E_SIGNAL	16
#define	E_INTERN	17
#define	E_EOF		18
#define E_SHARED_VECS   19
#define E_NEG           20
#define E_OVERWRITE     21
#define E_BREAKDOWN     22

/* warning types */
#define WARN_UNKNOWN     	0
#define WARN_WRONG_TYPE 	1
#define WARN_NO_MARK		2
#define WARN_RES_LESS_0         3
#define WARN_SHARED_VEC		4


/* error catching macros */

/* execute err_part if error errnum is raised while executing ok_part */
#define	catch(errnum,ok_part,err_part)	\
	{	jmp_buf _save;	int _err_num, _old_flag; \
		_old_flag = set_err_flag(EF_SILENT); \
		MEM_COPY(restart,_save,sizeof(jmp_buf)); \
		if ( (_err_num=setjmp(restart)) == 0 ) \
		{	ok_part; \
			set_err_flag(_old_flag); \
			MEM_COPY(_save,restart,sizeof(jmp_buf));	} \
		else if ( _err_num == errnum ) \
		{	set_err_flag(_old_flag);  \
			MEM_COPY(_save,restart,sizeof(jmp_buf)); \
			err_part;	} \
		else {	set_err_flag(_old_flag); \
			MEM_COPY(_save,restart,sizeof(jmp_buf)); \
			error(_err_num,"catch"); \
		} \
	}


/* execute err_part if any error raised while executing ok_part */
#define	catchall(ok_part,err_part) \
	{	jmp_buf _save;	int _err_num, _old_flag; \
		_old_flag = set_err_flag(EF_SILENT); \
		MEM_COPY(restart,_save,sizeof(jmp_buf)); \
		if ( (_err_num=setjmp(restart)) == 0 ) \
		{	ok_part; \
			set_err_flag(_old_flag); \
			MEM_COPY(_save,restart,sizeof(jmp_buf));	} \
		else \
		{	set_err_flag(_old_flag);  \
			MEM_COPY(_save,restart,sizeof(jmp_buf)); \
			err_part;	} \
	}


/* print message if error raised while executing ok_part,
                then re-raise error to trace calls */
#define	tracecatch(ok_part,function) \
	{	jmp_buf _save;	int _err_num, _old_flag; \
		_old_flag = set_err_flag(EF_JUMP); \
		MEM_COPY(restart,_save,sizeof(jmp_buf)); \
		if ( (_err_num=setjmp(restart)) == 0 ) \
		{	ok_part; \
			set_err_flag(_old_flag); \
			MEM_COPY(_save,restart,sizeof(jmp_buf));	} \
		else \
		{	set_err_flag(_old_flag);  \
			MEM_COPY(_save,restart,sizeof(jmp_buf)); \
			error(_err_num,function);	} \
	}



#endif   /* ERRHEADER */

