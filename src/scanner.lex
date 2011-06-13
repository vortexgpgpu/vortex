/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
%option c++
%option noyywrap

%{
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <string>
#include <iostream>

#include "include/asm-tokens.h"

static int64_t read_number(const char *s) {
  long long u;
  while (!isdigit(*s) && *s != '-' && *s != '+') s++;

  sscanf(s, "%lli", &u);
  return u;
}

static std::string label_name(const char *cs) {
  return std::string(cs, strlen(cs)-1);
}

struct rval_t { std::string s; uint64_t u; } yylval;
unsigned yyline(1);

using namespace HarpTools;

%}

%start DEFARGS PERMARGS WORDARGS STRINGARGS ALIGNARGS EMPTYARGS INSTARGS
%start EATCOLON

sym [A-Za-z_][A-Za-z0-9_]*
decnum [1-9][0-9]* 
hexnum 0x[0-9a-f]+
octnum 0[0-9]*
num [+-]?({decnum}|{hexnum}|{octnum})
space [ \t]*
peoperator ("+"|"-"|"*"|"/"|"%"|"&"|"|"|"^"|"<<"|">>")
parenexp "("({num}|{sym}|{peoperator}|{space}|"("|")")+")"
endl \r?\n

%%
\/\*([^*]|\*[^/]|{endl})*\*\/ {
  /* Ignore comments but keep line count consistent. */
  for (const char *c = YYText(); *c; c++) if (*c == '\n') yyline++;
}

<INITIAL>\.def    { BEGIN DEFARGS;    return ASM_T_DIR_DEF;   }
<INITIAL>\.perm   { BEGIN PERMARGS;   return ASM_T_DIR_PERM;  }
<INITIAL>\.byte   { BEGIN WORDARGS;   return ASM_T_DIR_BYTE;  }
<INITIAL>\.word   { BEGIN WORDARGS;   return ASM_T_DIR_WORD;  }
<INITIAL>\.string { BEGIN STRINGARGS; return ASM_T_DIR_STRING; }
<INITIAL>\.align  { BEGIN ALIGNARGS;  return ASM_T_DIR_ALIGN; }
<INITIAL>\.entry  { BEGIN EMPTYARGS;  return ASM_T_DIR_ENTRY; }
<INITIAL>\.global { BEGIN EMPTYARGS;  return ASM_T_DIR_GLOBAL; }

<INITIAL>@p{num}{space}\? { yylval.u = read_number(YYText());
                            return ASM_T_PRED; }
<INITIAL>{sym}/: { BEGIN EATCOLON;
                   yylval.s = std::string(YYText());
                   return ASM_T_LABEL; }
<INITIAL>{sym} { BEGIN INSTARGS;
                 yylval.s = std::string(YYText());
                 return ASM_T_INST; }
<INITIAL>; {}
<INITIAL>{endl} {yyline++;}

<EATCOLON>: { BEGIN INITIAL; }


<INSTARGS>@p{num}{space}[,;]? { yylval.u = read_number(YYText());
                                return ASM_T_PREG; }
<INSTARGS>%r{num}{space}[,;]? { yylval.u = read_number(YYText());
                                return ASM_T_REG; }
<INSTARGS>#{num}{space}[,;]? { yylval.u = read_number(YYText());
                               return ASM_T_LIT; }
<INSTARGS>{sym} { yylval.s = std::string(YYText()); return ASM_T_SYM; }
<INSTARGS>{parenexp} { yylval.s = std::string(YYText()); return ASM_T_PEXP; }

<INSTARGS>{space} {}
<INSTARGS>; { BEGIN INITIAL; return ASM_T_DIR_END; }
<INSTARGS>{endl} { BEGIN INITIAL; yyline++; return ASM_T_DIR_END; }


<DEFARGS>{sym} { yylval.s = std::string(YYText());
                 return ASM_T_DIR_ARG_SYM; }
<DEFARGS>{num} { yylval.u = read_number(YYText());
                 return ASM_T_DIR_ARG_NUM; }
<DEFARGS>{endl} { yyline++; BEGIN INITIAL; }


<PERMARGS>r      { return ASM_T_DIR_ARG_R; }
<PERMARGS>w      { return ASM_T_DIR_ARG_W; }
<PERMARGS>x      { return ASM_T_DIR_ARG_X; }
<PERMARGS>{endl} { BEGIN INITIAL; yyline++; return ASM_T_DIR_END; }


<WORDARGS>{sym}  { yylval.s = std::string(YYText());
                   return ASM_T_DIR_ARG_SYM; }
<WORDARGS>{num}  { yylval.u = read_number(YYText());
                   return ASM_T_DIR_ARG_NUM; }
<WORDARGS>{endl} { BEGIN INITIAL; yyline++; return ASM_T_DIR_END; }

<STRINGARGS>{sym} { yylval.s = std::string(YYText());
                    return ASM_T_DIR_ARG_SYM; }
<STRINGARGS>\"([^\"]|\\\")*\" { BEGIN INITIAL;
                                yylval.s = std::string(YYText());
                                yylval.s = yylval.s.substr(1,
                                             yylval.s.length() - 2);
                                return ASM_T_DIR_ARG_STRING; }

<ALIGNARGS>{num}  { yylval.u = read_number(YYText());
                    return ASM_T_DIR_ARG_NUM; }
<ALIGNARGS>{endl} { yyline++; BEGIN INITIAL; }


<EMPTYARGS>{endl} { yyline++; BEGIN INITIAL; }

{space} { /*Ignore inter-token whitespace.*/ }
. { std::cout << "Unexpected character on line " << std::dec << yyline << '\n';
     exit(1); }
