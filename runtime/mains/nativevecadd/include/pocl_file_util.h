/* pocl_file_util.h: global declarations of portable file utility functions
   defined in lib/llvmopencl, due to using llvm::sys::fs & other llvm APIs

   Copyright (c) 2015 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/


#ifndef POCL_FILE_UTIL_H
#define POCL_FILE_UTIL_H


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* Remove a directory, recursively */
int pocl_rm_rf(const char* path);

/* Make a directory, including all directories along path */
int pocl_mkdir_p(const char* path);

/* Remove a file or empty directory */
int pocl_remove(const char* path);

int pocl_rename(const char *oldpath, const char *newpath);

int pocl_exists(const char* path);

/* Touch file to change last modified time. For portability, this
 * removes & creates the file. */
int pocl_touch_file(const char* path);

/* Writes or appends data to a file.  */
int pocl_write_file(const char* path, const char* content,
                    uint64_t count, int append, int dont_rewrite);

int pocl_write_tempfile (char *output_path, const char *prefix,
                         const char *suffix, const char *content,
                         uint64_t count, int *ret_fd);

/* Allocates memory and places file contents in it.
 * Returns negative errno on error, zero otherwise. */
int pocl_read_file(const char* path, char** content, uint64_t *filesize);

int pocl_write_module(void *module, const char* path, int dont_rewrite);

int pocl_mk_tempdir (char *output, const char *prefix);

int pocl_mk_tempname (char *output, const char *prefix, const char *suffix,
                      int *ret_fd);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif


#endif
