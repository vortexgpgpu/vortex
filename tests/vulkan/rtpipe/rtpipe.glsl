/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Shared by every rtpipe stage: the payload, the record layout and the
 * launch row that selects each ray's scenario. Mirrors main.c. */

#define P_KIND     0   /* which shader ended the trace (K_*)           */
#define P_HITKIND  1
#define P_T        2   /* gl_HitTEXT                                     */
#define P_PRIM     3
#define P_INST     4
#define P_CUSTOM   5
#define P_GEOM     6
#define P_ATTR_U   7   /* hit attribute: barycentrics or the IS's pair   */
#define P_ATTR_V   8
#define P_RECORD   9   /* shaderRecordEXT word of the record that ran    */
#define P_OBJ_OX   10  /* gl_ObjectRayOriginEXT.x                        */
#define P_O2W_TX   11  /* gl_ObjectToWorldEXT[3].x                       */
#define P_W2O_XX   12  /* gl_WorldToObjectEXT[0].x                       */
#define P_TMIN     13
#define P_TMAX     14
#define P_FLAGS    15
#define P_LAUNCH   16
#define P_SEC_KIND 17  /* the secondary ray a closest-hit traced         */
#define P_SEC_T    18
#define P_SEC_CUST 19
#define P_CALL     20  /* the callable's answer                          */
#define P_AHS      21  /* any-hit invocations                            */
#define P_WDIR_Z   22
#define P_RGEN_REC 23
#define P_THIRD    24  /* the third ray a recursion-row closest-hit traced */
#define P_WORDS    25

#define K_CHIT0 1u
#define K_CHIT1 2u
#define K_CHIT2 3u
#define K_MISS0 4u
#define K_MISS1 5u

/* Launch rows. */
#define ROW_ALPHA     0u  /* non-opaque triangle, any-hit alpha test     */
#define ROW_TERMINATE 1u  /* any-hit terminates the ray                  */
#define ROW_PROC_OPQ  2u  /* opaque box: intersection, no any-hit        */
#define ROW_PROC      3u  /* non-opaque box: intersection then any-hit   */
#define ROW_RECURSE   4u  /* closest-hit traces a second ray             */
#define ROW_MISS0     5u  /* miss index 0, ray flags and interval        */
#define ROW_MISS1     6u  /* miss through miss index 1                   */
#define ROW_SBT_OFF   7u  /* sbtRecordOffset 4, forced opaque            */
#define ROW_TIE       8u  /* opaque and accepted triangles at one t      */
#define ROWS          9u

struct Payload {
   uint w[P_WORDS];
};
