/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**
 * **/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					No longer performs "validity" function to
 * analyze	**/
/**					compactness and separation crietria; instead
 * **/
/**					calculate root mean squared error.
 * **/
/**                                                                     **/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 1

#include "kmeans.h"
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern double wtime(void);

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
  char *help = "\nUsage: %s [switches] -i filename\n\n"
               "    -i filename      :file containing data to be clustered\n"
               "    -m max_nclusters :maximum number of clusters allowed    "
               "[default=5]\n"
               "    -n min_nclusters :minimum number of clusters allowed    "
               "[default=5]\n"
               "    -t threshold     :threshold value                       "
               "[default=0.001]\n"
               "    -l nloops        :iteration for each number of clusters "
               "[default=1]\n"
               "    -b               :input file is in binary format\n"
               "    -r               :calculate RMSE                        "
               "[default=off]\n"
               "    -o               :output cluster center coordinates     "
               "[default=off]\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
  int opt;
  extern char *optarg;
  char *filename = 0;
  float *buf;
  char line[1024];
  int isBinaryFile = 0;

  float threshold = 0.001; /* default value */
  int max_nclusters = 5;   /* default value */
  int min_nclusters = 5;   /* default value */
  int best_nclusters = 0;
  int nfeatures = 100;
  int npoints = 100;
  float len;

  float **features;
  float **cluster_centres = NULL;
  int i, j, index;
  int nloops = 1; /* default value */

  int isRMSE = 0;
  float rmse;

  int isOutput = 0;
  // float	cluster_timing, io_timing;

  /* obtain command line arguments and change appropriate options */
  while ((opt = getopt(argc, argv, "p:f:i:t:m:n:l:bro")) != EOF) {
    switch (opt) {
    case 'i':
      filename = optarg;
      break;
    case 'b':
      isBinaryFile = 1;
      break;
    case 't':
      threshold = atof(optarg);
      break;
    case 'm':
      max_nclusters = atoi(optarg);
      break;
    case 'n':
      min_nclusters = atoi(optarg);
      break;
    case 'f':
      nfeatures = atoi(optarg);
      break;
    case 'p':
      npoints = atoi(optarg);
      break;
    case 'r':
      isRMSE = 1;
      break;
    case 'o':
      isOutput = 1;
      break;
    case 'l':
      nloops = atoi(optarg);
      break;
    case '?':
      usage(argv[0]);
      break;
    default:
      usage(argv[0]);
      break;
    }
  }

  /* ============== I/O begin ==============*/
  /* get nfeatures and npoints */
  // io_timing = omp_get_wtime();

  if (filename) {
    if (isBinaryFile) { // Binary file input
      FILE *infile;
      if ((infile = fopen("100", "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
      }
      fread(&npoints, 1, sizeof(int), infile);
      fread(&nfeatures, 1, sizeof(int), infile);

      // allocate space for features[][] and read attributes of all objects
      buf = (float *)malloc(npoints * nfeatures * sizeof(float));
      features = (float **)malloc(npoints * sizeof(float *));
      features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
      for (i = 1; i < npoints; i++) {
        features[i] = features[i - 1] + nfeatures;
      }
      fread(buf, 1, npoints * nfeatures * sizeof(float), infile);
      fclose(infile);
    } else {
      FILE *infile;
      if ((infile = fopen("100", "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
      }
      while (fgets(line, 1024, infile) != NULL)
        if (strtok(line, " \t\n") != 0) {
          npoints++;
        }
      rewind(infile);
      while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
          // ignore the id (first attribute): nfeatures = 1;
          while (strtok(NULL, " ,\t\n") != NULL)
            nfeatures++;
          break;
        }
      }

      // allocate space for features[] and read attributes of all objects
      buf = (float *)malloc(npoints * nfeatures * sizeof(float));
      features = (float **)malloc(npoints * sizeof(float *));
      features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
      for (i = 1; i < npoints; i++)
        features[i] = features[i - 1] + nfeatures;
      rewind(infile);
      i = 0;
      while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL)
          continue;
        for (j = 0; j < nfeatures; j++) {
          buf[i] = atof(strtok(NULL, " ,\t\n"));
          i++;
        }
      }
      fclose(infile);
    }
  } else {
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    features = (float **)malloc(npoints * sizeof(float *));
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (i = 1; i < npoints; i++) {
      features[i] = features[i - 1] + nfeatures;
    }
    for (i = 0; i < npoints * nfeatures; ++i) {   
      buf[i] = (i % 64);
    }
  }

  // io_timing = omp_get_wtime() - io_timing;

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", npoints);
  printf("Number of features: %d\n", nfeatures);
  /* ============== I/O end ==============*/

  // error check for clusters
  if (npoints < min_nclusters) {
    printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
           min_nclusters, npoints);
    exit(0);
  }

  srand(7); /* seed for future random number generator */
  memcpy(
      features[0], buf,
      npoints * nfeatures *
          sizeof(
              float)); /* now features holds 2-dimensional array of features */
  free(buf);

  /* ======================= core of the clustering ===================*/

  // cluster_timing = omp_get_wtime();		/* Total clustering time */
  cluster_centres = NULL;
  index = cluster(npoints,       /* number of data points */
                  nfeatures,     /* number of features for each point */
                  features,      /* array: [npoints][nfeatures] */
                  min_nclusters, /* range of min to max number of clusters */
                  max_nclusters, threshold, /* loop termination factor */
                  &best_nclusters,  /* return: number between min and max */
                  &cluster_centres, /* return: [best_nclusters][nfeatures] */
                  &rmse,            /* Root Mean Squared Error */
                  isRMSE,           /* calculate RMSE */
                  nloops); /* number of iteration for each number of clusters */

  // cluster_timing = omp_get_wtime() - cluster_timing;

  /* =============== Command Line Output =============== */

  /* cluster center coordinates
     :displayed only for when k=1*/
  if ((min_nclusters == max_nclusters) && (isOutput == 1)) {
    printf("\n================= Centroid Coordinates =================\n");
    for (i = 0; i < max_nclusters; i++) {
      printf("%d:", i);
      for (j = 0; j < nfeatures; j++) {
        printf(" %.2f", cluster_centres[i][j]);
      }
      printf("\n\n");
    }
  }

  len = (float)((max_nclusters - min_nclusters + 1) * nloops);

  printf("Number of Iteration: %d\n", nloops);
  // printf("Time for I/O: %.5fsec\n", io_timing);
  // printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

  if (min_nclusters != max_nclusters) {
    if (nloops != 1) { // range of k, multiple iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    } else { // range of k, single iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    }
  } else {
    if (nloops != 1) { // single k, multiple iteration
      // printf("Average Clustering Time: %.5fsec\n",
      //		cluster_timing / nloops);
      if (isRMSE) // if calculated RMSE
        printf("Number of trials to approach the best RMSE of %.3f is %d\n",
               rmse, index + 1);
    } else {      // single k, single iteration
      if (isRMSE) // if calculated RMSE
        printf("Root Mean Squared Error: %.3f\n", rmse);
    }
  }

  /* free up memory */  
  free(cluster_centres[0]);
  free(cluster_centres);
  free(features[0]);
  free(features);
  return (0);
}
