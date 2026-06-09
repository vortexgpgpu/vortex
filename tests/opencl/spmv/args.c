
#include <parboil.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*****************************************************************************/
/* Memory management routines */

/* Free an array of owned strings. */
void
pb_FreeStringArray(char **string_array)
{
  char **p;

  if (!string_array) return;
  for (p = string_array; *p; p++) free(*p);
  free(string_array);
}

struct pb_PlatformParam *
pb_PlatformParam(char *name, char *version)
{
  if (name == NULL) {
    fprintf(stderr, "pb_PlatformParam: Invalid argument\n");
    exit(-1);
  }

  struct pb_PlatformParam *ret =
    (struct pb_PlatformParam *)malloc(sizeof (struct pb_PlatformParam));

  ret->name = name;
  ret->version = version;
  return ret;
}

void
pb_FreePlatformParam(struct pb_PlatformParam *p)
{
  if (p == NULL) return;

  free(p->name);
  free(p->version);
  free(p);
}

struct pb_DeviceParam *
pb_DeviceParam_index(int index)
{
  struct pb_DeviceParam *ret =
    (struct pb_DeviceParam *)malloc(sizeof (struct pb_DeviceParam));
  ret->criterion = pb_Device_INDEX;
  ret->index = index;
  return ret;
}

struct pb_DeviceParam *
pb_DeviceParam_cpu(void)
{
  struct pb_DeviceParam *ret =
    (struct pb_DeviceParam *)malloc(sizeof (struct pb_DeviceParam));
  ret->criterion = pb_Device_CPU;
  return ret;
}

struct pb_DeviceParam *
pb_DeviceParam_gpu(void)
{
  struct pb_DeviceParam *ret =
    (struct pb_DeviceParam *)malloc(sizeof (struct pb_DeviceParam));
  ret->criterion = pb_Device_GPU;
  return ret;
}

struct pb_DeviceParam *
pb_DeviceParam_accelerator(void)
{
  struct pb_DeviceParam *ret =
    (struct pb_DeviceParam *)malloc(sizeof (struct pb_DeviceParam));
  ret->criterion = pb_Device_ACCELERATOR;
  return ret;
}

struct pb_DeviceParam *
pb_DeviceParam_name(char *name)
{
  struct pb_DeviceParam *ret =
    (struct pb_DeviceParam *)malloc(sizeof (struct pb_DeviceParam));
  ret->criterion = pb_Device_NAME;
  ret->name = name;
  return ret;
}

void
pb_FreeDeviceParam(struct pb_DeviceParam *p)
{
  if (p == NULL) return;

  switch(p->criterion) {
  case pb_Device_NAME:
    free(p->name);
    break;
  case pb_Device_INDEX:
  case pb_Device_CPU:
  case pb_Device_ACCELERATOR:
    break;
  default:
    fprintf(stderr, "pb_FreeDeviceParam: Invalid argument\n");
    exit(-1);
  }
}

void
pb_FreeParameters(struct pb_Parameters *p)
{
  free(p->outFile);
  pb_FreeStringArray(p->inpFiles);
  pb_FreePlatformParam(p->platform);
  pb_FreeDeviceParam(p->device);
  free(p);
}

/*****************************************************************************/

/* Parse a comma-delimited list of strings into an
 * array of strings. */
static char ** 
read_string_array(char *in)
{
  char **ret;
  int i;
  int count;			/* Number of items in the input */
  char *substring;		/* Current substring within 'in' */

  /* Count the number of items in the string */
  count = 1;
  for (i = 0; in[i]; i++) if (in[i] == ',') count++;

  /* Allocate storage */
  ret = (char **)malloc((count + 1) * sizeof(char *));

  /* Create copies of the strings from the list */
  substring = in;
  for (i = 0; i < count; i++) {
    char *substring_end;
    int substring_length;

    /* Find length of substring */
    for (substring_end = substring;
	 (*substring_end != ',') && (*substring_end != 0);
	 substring_end++);

    substring_length = substring_end - substring;

    /* Allocate memory and copy the substring */
    ret[i] = (char *)malloc(substring_length + 1);
    memcpy(ret[i], substring, substring_length);
    ret[i][substring_length] = 0;

    /* go to next substring */
    substring = substring_end + 1;
  }
  ret[i] = NULL;		/* Write the sentinel value */

  return ret;
}

static void
report_parse_error(const char *str)
{
  fputs(str, stderr);
}

/* Interpret a string as a 'pb_DeviceParam' value.
 * Return a pointer to a new value, or NULL on failure.
 */
static struct pb_DeviceParam *
read_device_param(char *str)
{  
  /* Try different ways of interpreting 'device_string' until one works */

  /* If argument is an integer, then interpret it as a device index */
  errno = 0;
  char *end;
  long device_int = strtol(str, &end, 10);
  if (!errno) {
    /* Negative numbers are not valid */
    if (device_int < 0 || device_int > INT_MAX) return NULL;

    return pb_DeviceParam_index(device_int);
  }

  /* Match against predefined strings */
  if (strcmp(str, "CPU") == 0)
    return pb_DeviceParam_cpu();
  if (strcmp(str, "GPU") == 0)
    return pb_DeviceParam_gpu();
  if (strcmp(str, "ACCELERATOR") == 0)
    return pb_DeviceParam_accelerator();

  /* Assume any other string is a device name */
  return pb_DeviceParam_name(strdup(str));
}

/* Interpret a string as a 'pb_PlatformParam' value.
 * Return a pointer to a new value, or NULL on failure.
 */
static struct pb_PlatformParam *
read_platform_param(char *str)
{
  int separator_index;          /* Index of the '-' character separating
                                 * name and version number.  It's -1 if
                                 * there's no '-' character. */

  /* Find the last occurrence of '-' in 'str' */
  {
    char *cur;
    separator_index = -1;
    for (cur = str; *cur; cur++) {
      if (*cur == '-') separator_index = cur - str;
    }
  }

  /* The platform name is either the entire string, or all characters before
   * the separator */
  int name_length = separator_index == -1 ? strlen(str) : separator_index;
  char *name_str = (char *)malloc(name_length + 1);
  memcpy(name_str, str, name_length);
  name_str[name_length] = 0;

  /* The version is either NULL, or all characters after the separator */
  char *version_str;
  if (separator_index == -1) {
    version_str = NULL;
  }
  else {
    const char *version_input_str = str + separator_index + 1;
    int version_length = strlen(version_input_str);

    version_str = (char *)malloc(version_length + 1);
    memcpy(version_str, version_input_str, version_length);
    version_str[version_length] = 0;
  }

  /* Create output structure */
  return pb_PlatformParam(name_str, version_str);
}

/****************************************************************************/
/* Argument parsing state */

/* Argument parsing state.
 *
 * Arguments that are interpreted by the argument parser are removed from
 * the list.  Variables 'argc' and 'argn' do not count arguments that have
 * been removed.
 *
 * During argument parsing, the array of arguments is compacted, overwriting
 * the erased arguments.  Variable 'argv_put' points to the array element
 * where the next argument will be written.  Variable 'argv_get' points to
 * the array element where the next argument will be read from.
 */
struct argparse {
  int argc;			/* Number of arguments.  Mutable. */
  int argn;			/* Current argument index. */
  char **argv_get;		/* Argument value being read. */
  char **argv_put;		/* Argument value being written.
				 * argv_put <= argv_get. */
};

static void
initialize_argparse(struct argparse *ap, int argc, char **argv)
{
  ap->argc = argc;
  ap->argn = 0;
  ap->argv_get = ap->argv_put = argv;
}

/* Finish argument parsing, without processing the remaining arguments.
 * Write new argument count into _argc. */
static void
finalize_argparse(struct argparse *ap, int *_argc, char **argv)
{
  /* Move the remaining arguments */
  for(; ap->argn < ap->argc; ap->argn++)
    *ap->argv_put++ = *ap->argv_get++;

  /* Update the argument count */
  *_argc = ap->argc;

  /* Insert a terminating NULL */
  argv[ap->argc] = NULL;
}

/* Delete the current argument.  The argument will not be visible
 * when argument parsing is done. */
static void
delete_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "delete_argument\n");
  }
  ap->argc--;
  ap->argv_get++;
}

/* Go to the next argument.  Also, move the current argument to its
 * final location in argv. */
static void
next_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "next_argument\n");
  }
  /* Move argument to its new location. */
  *ap->argv_put++ = *ap->argv_get++;
  ap->argn++;
}

static int
is_end_of_arguments(struct argparse *ap)
{
  return ap->argn == ap->argc;
}

/* Get the current argument */
static char *
get_argument(struct argparse *ap)
{
  return *ap->argv_get;
}

/* Get the current argument, and also delete it */
static char *
consume_argument(struct argparse *ap)
{
  char *ret = get_argument(ap);
  delete_argument(ap);
  return ret;
}

/****************************************************************************/

/* The result of parsing a command-line argument */
typedef enum {
  ARGPARSE_OK,                  /* Success */
  ARGPARSE_ERROR,               /* Error */
  ARGPARSE_DONE                 /* Success, and do not continue parsing */
} result;

typedef result parse_action(struct argparse *ap, struct pb_Parameters *params);


/* A command-line option */
struct option {
  char short_name;              /* If not 0, the one-character
                                 * name of this option */
  const char *long_name;        /* If not NULL, the long name of this option */
  parse_action *action;       /* What to do when this option occurs.
                               * Sentinel value is NULL.
                               */
};

/* Output file
 *
 * -o FILE
 */
static result
parse_output_file(struct argparse *ap, struct pb_Parameters *params)
{
  if (is_end_of_arguments(ap))
    {
      report_parse_error("Expecting file name after '-o'\n");
      return ARGPARSE_ERROR;
    }

  /* Replace the output file name */
  free(params->outFile);
  params->outFile = strdup(consume_argument(ap));

  return ARGPARSE_OK;
}

/* Input files
 *
 * -i FILE,FILE,...
 */
static result
parse_input_files(struct argparse *ap, struct pb_Parameters *params)
{
  if (is_end_of_arguments(ap))
    {
      report_parse_error("Expecting file name after '-i'\n");
      return ARGPARSE_ERROR;
    }

  /* Replace the input file list */
  pb_FreeStringArray(params->inpFiles);
  params->inpFiles = read_string_array(consume_argument(ap));
  return ARGPARSE_OK;
}

/* End of options
 *
 * --
 */

static result
parse_end_options(struct argparse *ap, struct pb_Parameters *params)
{
  return ARGPARSE_DONE;
}

/* OpenCL device
 *
 * --device X
 */

static result
parse_device(struct argparse *ap, struct pb_Parameters *params)
{
  /* Read the next argument, which specifies a device */

  if (is_end_of_arguments(ap))
    {
      report_parse_error("Expecting device specification after '--device'\n");
      return ARGPARSE_ERROR;
    }

  char *device_string = consume_argument(ap);
  struct pb_DeviceParam *device_param = read_device_param(device_string);

  if (!device_param) {
    report_parse_error("Unrecognized device specification format on command line\n");
    return ARGPARSE_ERROR;
  }

  /* Save the result */
  pb_FreeDeviceParam(params->device);
  params->device = device_param;

  return ARGPARSE_OK;
}

static result
parse_platform(struct argparse *ap, struct pb_Parameters *params)
{
  /* Read the next argument, which specifies a platform */

  if (is_end_of_arguments(ap))
    {
      report_parse_error("Expecting device specification after '--platform'\n");
      return ARGPARSE_ERROR;
    }

  char *platform_string = consume_argument(ap);
  struct pb_PlatformParam *platform_param = read_platform_param(platform_string);

  if (!platform_param) {
    report_parse_error("Unrecognized platform specification format on command line\n");
    return ARGPARSE_ERROR;
  }

  /* Save the result */
  pb_FreePlatformParam(params->platform);
  params->platform = platform_param;

  return ARGPARSE_OK;
}
  

static struct option options[] = {
  { 'o', NULL, &parse_output_file },
  { 'i', NULL, &parse_input_files },
  { '-', NULL, &parse_end_options },
  { 0, "device", &parse_device },
  { 0, "platform", &parse_platform },
  { 0, NULL, NULL }
};

static int
is_last_option(struct option *op)
{
  return op->action == NULL;
}

/****************************************************************************/

/* Parse command-line parameters.
 * Return zero on error, nonzero otherwise.
 * On error, the other outputs may be invalid.
 *
 * The information collected from parameters is used to update
 * 'ret'.  'ret' should be initialized.
 *
 * '_argc' and 'argv' are updated to contain only the unprocessed arguments.
 */
static int
pb_ParseParameters (struct pb_Parameters *ret, int *_argc, char **argv)
{
  char *err_message;
  struct argparse ap;

  /* Each argument */
  initialize_argparse(&ap, *_argc, argv);
  while(!is_end_of_arguments(&ap)) {
    result arg_result;          /* Result of parsing this option */
    char *arg = get_argument(&ap);

    /* Process this argument */
    if (arg[0] == '-') {
      /* Single-character flag */
      if ((arg[1] != 0) && (arg[2] == 0)) {
        delete_argument(&ap);	/* This argument is consumed here */

        /* Find a matching short option */
        struct option *op;
        for (op = options; !is_last_option(op); op++) {
          if (op->short_name == arg[1]) {
            arg_result = (*op->action)(&ap, ret);
            goto option_was_processed;
          }
        }

        /* No option matches */
        report_parse_error("Unexpected command-line parameter\n");
        arg_result = ARGPARSE_ERROR;
        goto option_was_processed;
      }

      /* Long flag */
      if (arg[1] == '-') {
        delete_argument(&ap);	/* This argument is consumed here */

        /* Find a matching long option */
        struct option *op;
        for (op = options; !is_last_option(op); op++) {
          if (op->long_name && strcmp(&arg[2], op->long_name) == 0) {
            arg_result = (*op->action)(&ap, ret);
            goto option_was_processed;
          }
        }

        /* No option matches */
        report_parse_error("Unexpected command-line parameter\n");
        arg_result = ARGPARSE_ERROR;
        goto option_was_processed;
      }
    }
    else {
      /* Other arguments are ignored */
      next_argument(&ap);
      arg_result = ARGPARSE_OK;
      goto option_was_processed;
    }

  option_was_processed:
    /* Decide what to do next based on 'arg_result' */
    switch(arg_result) {
    case ARGPARSE_OK:
      /* Continue processing */
      break;

    case ARGPARSE_ERROR:
      /* Error exit from the function */
      return 0;

    case ARGPARSE_DONE:
      /* Normal exit from the argument parsing loop */
      goto end_of_options;
    }
  } /* end for each argument */

  /* If all arguments were processed, then normal exit from the loop */

 end_of_options:
  finalize_argparse(&ap, _argc, argv);
  return 1;
}

/*****************************************************************************/
/* Other exported functions */

struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv)
{
  struct pb_Parameters *ret =
    (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));

  /* Initialize the parameters structure */
  ret->outFile = NULL;
  ret->inpFiles = (char **)malloc(sizeof(char *));
  ret->inpFiles[0] = NULL;
  ret->platform = NULL;
  ret->device = NULL;

  /* Read parameters and update _argc, argv */
  if (!pb_ParseParameters(ret, _argc, argv)) {
    /* Parse error */
    pb_FreeParameters(ret);
    return NULL;
  }

  return ret;
}

int
pb_Parameters_CountInputs(struct pb_Parameters *p)
{
  int n;

  for (n = 0; p->inpFiles[n]; n++);
  return n;
}

