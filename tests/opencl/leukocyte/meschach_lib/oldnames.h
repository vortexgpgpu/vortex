
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


/* macros for names used in versions 1.0 and 1.1 */
/* 8/11/93 */


#ifndef OLDNAMESH
#define OLDNAMESH


/* type IVEC */

#define get_ivec   iv_get
#define freeivec   IV_FREE
#define cp_ivec    iv_copy
#define fout_ivec  iv_foutput
#define out_ivec   iv_output
#define fin_ivec   iv_finput
#define in_ivec    iv_input
#define dump_ivec  iv_dump


/* type ZVEC */

#define get_zvec   zv_get
#define freezvec   ZV_FREE
#define cp_zvec    zv_copy
#define fout_zvec  zv_foutput
#define out_zvec   zv_output
#define fin_zvec   zv_finput
#define in_zvec    zv_input
#define zero_zvec  zv_zero
#define rand_zvec  zv_rand
#define dump_zvec  zv_dump

/* type ZMAT */

#define get_zmat   zm_get
#define freezmat   ZM_FREE
#define cp_zmat    zm_copy
#define fout_zmat  zm_foutput
#define out_zmat   zm_output
#define fin_zmat   zm_finput
#define in_zmat    zm_input
#define zero_zmat  zm_zero
#define rand_zmat  zm_rand
#define dump_zmat  zm_dump

/* types SPMAT */

#define sp_mat        SPMAT
#define sp_get_mat    sp_get
#define sp_free_mat   sp_free
#define sp_cp_mat     sp_copy
#define sp_cp_mat2    sp_copy2
#define sp_fout_mat   sp_foutput
#define sp_fout_mat2  sp_foutput2
#define sp_out_mat    sp_output
#define sp_out_mat2   sp_output2
#define sp_fin_mat    sp_finput
#define sp_in_mat     sp_input
#define sp_zero_mat   sp_zero
#define sp_dump_mat   sp_dump


/* type SPROW */

#define sp_row        SPROW
#define sp_get_idx    sprow_idx
#define row_xpd       sprow_xpd
#define sp_get_row    sprow_get
#define row_set_val   sprow_set_val
#define fout_row      sprow_foutput
#define _row_mltadd   sprow_mltadd
#define sp_row_copy   sprow_copy
#define sp_row_merge  sprow_merge
#define sp_row_ip     sprow_ip
#define sp_row_sqr    sprow_sqr


/* type MAT */

#define get_mat   m_get
#define freemat   M_FREE
#define cp_mat    m_copy
#define fout_mat  m_foutput
#define out_mat   m_output
#define fin_mat   m_finput
#define in_mat    m_input
#define zero_mat  m_zero
#define id_mat    m_ident
#define rand_mat  m_rand
#define ones_mat  m_ones
#define dump_mat  m_dump

/* type VEC */

#define get_vec   v_get
#define freevec   V_FREE
#define cp_vec    v_copy
#define fout_vec  v_foutput
#define out_vec   v_output
#define fin_vec   v_finput
#define in_vec    v_input
#define zero_vec  v_zero
#define rand_vec  v_rand
#define ones_vec  v_ones
#define dump_vec  v_dump


/* type PERM */

#define get_perm   px_get
#define freeperm   PX_FREE
#define cp_perm    px_copy
#define fout_perm  px_foutput
#define out_perm   px_output
#define fin_perm   px_finput
#define in_perm    px_input
#define id_perm    px_ident
#define px_id      px_ident
#define trans_px   px_transp
#define sign_px    px_sign
#define dump_perm  px_dump

#endif
