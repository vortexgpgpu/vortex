/*
 * hurricanegen.c
 * Original author unknown
 * Modified by Sam Kauffman - University of Virginia
 *
 * Generates datasets of "hurricanes" to be used by Rodinia's Nearest Neighbor (nn)
 * Also generates lists of the files in the dataset. These lists are passed to nn.
 *
 * Usage: hurricanegen <num_hurricanes> <num_files>
 * The number of hurricanes should be a multiple of both 1024 and the number of files.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 641986 gets you ~30 MB of data
int main(int argc, char **argv)
{
	FILE *fp;
	int i = 0, total_canes = 0, canes = 0, num_files = 0, j = 0;
	int year, month, date, hour, num, speed, press;
	float lat, lon;
	int hours[4] =
	{ 0, 6, 12, 18 };
	char *name, fname[30];
	char names[21][10] =
	{ "ALBERTO", "BERYL", "CHRIS", "DEBBY", "ERNESTO", "FLORENCE", "GORDON",
			"HELENE", "ISAAC", "JOYCE", "KIRK", "LESLIE", "MICHAEL", "NADINE",
			"OSCAR", "PATTY", "RAFAEL", "SANDY", "TONY", "VALERIE", "WILLIAM" };

	if (argc < 3)
	{
		fprintf(stderr, "Error: Enter a number of hurricanes and a number of files.\n");
		fprintf(stderr, "The number of hurricanes should be a multiple of both 1024\nand the number of files.\n");
		exit(0);
	}

	total_canes = atoi(argv[1]);
	num_files = atoi(argv[2]);

	total_canes = ((total_canes+1023)/1024) * 1024; // round up to multiple of 1024
	canes = (total_canes + num_files - 1) / num_files; // round up (ceiling division)
	total_canes = canes * num_files;

	srand(time(NULL));

	for (j = 0; j < num_files; j++)
	{
		if (num_files == 1)
			sprintf(fname, "cane%dk.db", total_canes / 1024);
		else
			sprintf(fname, "cane%dk_%d_%d.db", total_canes / 1024, num_files, j);

		if ((fp = fopen(fname, "w")) == NULL)
		{
			fprintf(stderr, "Failed to open output file '%s'!\n", fname);
			return -1;
		}

		for (i = 0; i < canes; i++)
		{
			year = 1950 + rand() % 55;
			month = 1 + rand() % 12;
			date = 1 + rand() % 28;
			hour = hours[rand() % 4];
			num = 1 + rand() % 28;
			name = names[rand() % 21];
			lat = ((float) (7 + rand() % 63))
					+ ((float) rand() / (float) 0x7fffffff);
			lon = ((float) (rand() % 358))
					+ ((float) rand() / (float) 0x7fffffff);
			speed = 10 + rand() % 155;
			press = rand() % 900;

			fprintf(fp, "%4d %2d %2d %2d %2d %-9s %5.1f %5.1f %4d %4d\n",
					year, month, date, hour, num, name, lat, lon, speed, press);

		}

		fclose(fp);
	}
	printf("Generated %d hurricanes in %d file(s).\n", total_canes, num_files);

	if (num_files == 1)
	{
		sprintf(fname, "list%dk.txt", total_canes / 1024);
		fp = fopen(fname, "w");

		fprintf(fp, "../../data/nn/cane%dk.db\n", total_canes / 1024);
	}
	else
	{
		sprintf(fname, "list%dk_%d.txt", total_canes / 1024, num_files);
		fp = fopen(fname, "w");

		for (int i = 0; i < num_files; i++)
			fprintf(fp, "../../data/nn/cane%dk_%d_%d.db\n", total_canes / 1024, num_files, i);
	}

	fclose(fp);
	printf( "File list written to %s.\n", fname );
	return 0;
}