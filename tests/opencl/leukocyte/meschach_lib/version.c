
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


/*			Version routine			*/
/*	This routine must be modified whenever modifications are made to
	Meschach by persons other than the original authors
	(David E. Stewart & Zbigniew Leyk); 
	when new releases of Meschach are made the
	version number will also be updated
*/

#include	<stdio.h>

void	m_version()
{
	static char rcsid[] = "$Id: version.c,v 1.9 1994/03/24 00:04:05 des Exp $";

	printf("Meshach matrix library version 1.2b\n");
	printf("RCS id: %s\n",rcsid);
	printf("Changes since 1.2a:\n");
	printf("\t Fixed bug in schur() for 2x2 blocks with real e-vals\n");
	printf("\t Fixed bug in schur() reading beyond end of array\n");
	printf("\t Fixed some installation bugs\n");
	printf("\t Fixed bugs & improved efficiency in spILUfactor()\n");
	printf("\t px_inv() doesn't crash inverting non-permutations\n");
	printf("\t Fixed bug in ifft()\n");
	/**** List of modifications ****/
	/* Example below is for illustration only */
	/* printf("Modified by %s, routine(s) %s, file %s on date %s\n",
			"Joe Bloggs",
			"m_version",
			"version.c",
			"Fri Apr  5 16:00:38 EST 1994"); */
	/* printf("Purpose: %s\n",
			"To update the version number"); */
}

/* $Log: version.c,v $
 * Revision 1.9  1994/03/24  00:04:05  des
 * Added notes on changes to spILUfactor() and px_inv().
 *
 * Revision 1.8  1994/02/21  04:32:25  des
 * Set version to 1.2b with bug fixes in schur() and installation.
 *
 * Revision 1.7  1994/01/13  05:43:57  des
 * Version 1.2 update
 *

 * */
