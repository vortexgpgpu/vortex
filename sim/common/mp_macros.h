// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// macro primitives

#define MP_COMMA ,
#define MP_REM(...) __VA_ARGS__
#define MP_EAT(...)

#define MP_STRINGIZE_(x) #x
#define MP_STRINGIZE(x) MP_STRINGIZE_(x)

#define MP_CONCAT_(x, ...) x ## __VA_ARGS__
#define MP_CONCAT(x, ...) MP_CONCAT_(x, __VA_ARGS__)

#define MP_COUNTOF(arr) (sizeof(arr) / sizeof(arr[0]))

// conditional macro

#define MP_IIF_0(x, y) y
#define MP_IIF_1(x, y) x
#define MP_IIF(c) MP_CONCAT(MP_IIF_, c)

#define MP_PAIR_FIRST(a, b) a
#define MP_PAIR_SECOND(a, b) b

// pair macros

#define MP_PAIR(x) MP_REM x
#define MP_PAIR_HEAD_(x, ...) MP_PAIR(x)
#define MP_PAIR_PROBE_(...) (__VA_ARGS__),
#define MP_PAIR_L_(...) MP_PAIR_HEAD_(__VA_ARGS__)
#define MP_PAIR_L(x) MP_PAIR_L_(MP_PAIR_PROBE_ x,)
#define MP_PAIR_R(x) MP_EAT x

// separator macros

#define MP_SEP_COMMA() ,
#define MP_SEP_SEMICOLON() ;
#define MP_SEP_PLUS() +
#define MP_SEP_AND() &
#define MP_SEP_OR() |
#define MP_SEP_COLON() :
#define MP_SEP_SPACE() /**/
#define MP_SEP_LESS() <
#define MP_SEP_GREATER() >
#define MP_SEP_ANDL() &&
#define MP_SEP_ORL() ||

// MAKE_UNIQUE macro

#define MP_MAKE_UNIQUE(x) MP_CONCAT(x, __COUNTER__)

// increment macro

#define MP_INC(x) MP_INC_ ## x
#define MP_INC_0 1
#define MP_INC_1 2
#define MP_INC_2 3
#define MP_INC_3 4
#define MP_INC_4 5
#define MP_INC_5 6
#define MP_INC_6 7
#define MP_INC_7 8
#define MP_INC_8 9
#define MP_INC_9 10
#define MP_INC_10 11
#define MP_INC_11 12
#define MP_INC_12 13
#define MP_INC_13 14
#define MP_INC_14 15
#define MP_INC_15 16
#define MP_INC_16 17
#define MP_INC_17 18
#define MP_INC_18 19
#define MP_INC_19 20
#define MP_INC_20 21
#define MP_INC_21 22
#define MP_INC_22 23
#define MP_INC_23 24
#define MP_INC_24 25
#define MP_INC_25 26
#define MP_INC_26 27
#define MP_INC_27 28
#define MP_INC_28 29
#define MP_INC_29 30
#define MP_INC_30 31
#define MP_INC_31 32
#define MP_INC_32 33
#define MP_INC_33 34
#define MP_INC_34 35
#define MP_INC_35 36
#define MP_INC_36 37
#define MP_INC_37 38
#define MP_INC_38 39
#define MP_INC_39 40
#define MP_INC_40 41
#define MP_INC_41 42
#define MP_INC_42 43
#define MP_INC_43 44
#define MP_INC_44 45
#define MP_INC_45 46
#define MP_INC_46 47
#define MP_INC_47 48
#define MP_INC_48 49
#define MP_INC_49 50
#define MP_INC_50 51
#define MP_INC_51 52
#define MP_INC_52 53
#define MP_INC_53 54
#define MP_INC_54 55
#define MP_INC_55 56
#define MP_INC_56 57
#define MP_INC_57 58
#define MP_INC_58 59
#define MP_INC_59 60
#define MP_INC_60 61
#define MP_INC_61 62
#define MP_INC_62 63
#define MP_INC_63 64

// NARG macro

#define MP_NARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10,_11,_12,_13,_14,_15,_16, \
                  _17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32, \
                  _33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48, \
                  _49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63, N, ...) N

#define MP_NARG_R() 63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48, \
                    47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32, \
                    31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16, \
                    15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define MP_NARG_(...) MP_NARG_N(__VA_ARGS__)
#define MP_NARG(...)  MP_NARG_(__VA_ARGS__, MP_NARG_R())

// FOR_EACH macro

#define MP_FOR_EACH_1(idx, func, arg, sep, ...)      func(arg, idx, __VA_ARGS__)
#define MP_FOR_EACH_2(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_1(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_3(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_2(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_4(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_3(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_5(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_4(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_6(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_5(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_7(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_6(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_8(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_7(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_9(idx, func, arg, sep, x, ...)   func(arg, idx, x) sep() MP_FOR_EACH_8(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_10(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_9(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_11(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_10(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_12(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_11(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_13(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_12(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_14(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_13(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_15(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_14(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_16(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_15(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_17(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_16(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_18(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_17(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_19(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_18(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_20(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_19(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_21(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_20(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_22(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_21(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_23(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_22(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_24(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_23(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_25(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_24(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_26(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_25(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_27(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_26(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_28(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_27(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_29(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_28(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_30(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_29(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_31(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_30(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_32(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_31(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_33(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_32(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_34(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_33(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_35(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_34(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_36(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_35(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_37(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_36(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_38(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_37(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_39(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_38(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_40(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_39(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_41(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_40(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_42(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_41(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_43(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_42(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_44(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_43(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_45(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_44(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_46(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_45(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_47(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_46(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_48(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_47(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_49(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_48(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_50(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_49(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_51(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_50(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_52(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_51(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_53(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_52(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_54(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_53(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_55(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_54(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_56(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_55(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_57(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_56(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_58(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_57(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_59(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_58(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_60(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_59(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_61(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_60(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_62(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_61(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_63(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_62(MP_INC(idx), func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH_64(idx, func, arg, sep, x, ...)  func(arg, idx, x) sep() MP_FOR_EACH_63(MP_INC(idx), func, arg, sep, __VA_ARGS__)

#define MP_FOR_EACH_(N, func, arg, sep, ...) MP_CONCAT(MP_FOR_EACH_, N)(0, func, arg, sep, __VA_ARGS__)
#define MP_FOR_EACH(func, arg, sep, ...) MP_FOR_EACH_(MP_NARG(__VA_ARGS__), func, arg, sep, __VA_ARGS__)

// REVERSE_FOR_EACH macro

#define MP_REVERSE_FOR_EACH_1(func, arg, sep, ...)      func(arg, 0, __VA_ARGS__)
#define MP_REVERSE_FOR_EACH_2(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_1(func, arg, sep, __VA_ARGS__) sep() func(arg, 1, x)
#define MP_REVERSE_FOR_EACH_3(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_2(func, arg, sep, __VA_ARGS__) sep() func(arg, 2, x)
#define MP_REVERSE_FOR_EACH_4(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_3(func, arg, sep, __VA_ARGS__) sep() func(arg, 3, x)
#define MP_REVERSE_FOR_EACH_5(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_4(func, arg, sep, __VA_ARGS__) sep() func(arg, 4, x)
#define MP_REVERSE_FOR_EACH_6(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_5(func, arg, sep, __VA_ARGS__) sep() func(arg, 5, x)
#define MP_REVERSE_FOR_EACH_7(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_6(func, arg, sep, __VA_ARGS__) sep() func(arg, 6, x)
#define MP_REVERSE_FOR_EACH_8(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_7(func, arg, sep, __VA_ARGS__) sep() func(arg, 7, x)
#define MP_REVERSE_FOR_EACH_9(func, arg, sep, x, ...)   MP_REVERSE_FOR_EACH_8(func, arg, sep, __VA_ARGS__) sep() func(arg, 8, x)
#define MP_REVERSE_FOR_EACH_10(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_9(func, arg, sep, __VA_ARGS__) sep() func(arg, 9, x)
#define MP_REVERSE_FOR_EACH_11(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_10(func, arg, sep, __VA_ARGS__) sep() func(arg, 10, x)
#define MP_REVERSE_FOR_EACH_12(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_11(func, arg, sep, __VA_ARGS__) sep() func(arg, 11, x)
#define MP_REVERSE_FOR_EACH_13(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_12(func, arg, sep, __VA_ARGS__) sep() func(arg, 12, x)
#define MP_REVERSE_FOR_EACH_14(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_13(func, arg, sep, __VA_ARGS__) sep() func(arg, 13, x)
#define MP_REVERSE_FOR_EACH_15(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_14(func, arg, sep, __VA_ARGS__) sep() func(arg, 14, x)
#define MP_REVERSE_FOR_EACH_16(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_15(func, arg, sep, __VA_ARGS__) sep() func(arg, 15, x)
#define MP_REVERSE_FOR_EACH_17(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_16(func, arg, sep, __VA_ARGS__) sep() func(arg, 16, x)
#define MP_REVERSE_FOR_EACH_18(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_17(func, arg, sep, __VA_ARGS__) sep() func(arg, 17, x)
#define MP_REVERSE_FOR_EACH_19(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_18(func, arg, sep, __VA_ARGS__) sep() func(arg, 18, x)
#define MP_REVERSE_FOR_EACH_20(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_19(func, arg, sep, __VA_ARGS__) sep() func(arg, 19, x)
#define MP_REVERSE_FOR_EACH_21(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_20(func, arg, sep, __VA_ARGS__) sep() func(arg, 20, x)
#define MP_REVERSE_FOR_EACH_22(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_21(func, arg, sep, __VA_ARGS__) sep() func(arg, 21, x)
#define MP_REVERSE_FOR_EACH_23(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_22(func, arg, sep, __VA_ARGS__) sep() func(arg, 22, x)
#define MP_REVERSE_FOR_EACH_24(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_23(func, arg, sep, __VA_ARGS__) sep() func(arg, 23, x)
#define MP_REVERSE_FOR_EACH_25(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_24(func, arg, sep, __VA_ARGS__) sep() func(arg, 24, x)
#define MP_REVERSE_FOR_EACH_26(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_25(func, arg, sep, __VA_ARGS__) sep() func(arg, 25, x)
#define MP_REVERSE_FOR_EACH_27(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_26(func, arg, sep, __VA_ARGS__) sep() func(arg, 26, x)
#define MP_REVERSE_FOR_EACH_28(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_27(func, arg, sep, __VA_ARGS__) sep() func(arg, 27, x)
#define MP_REVERSE_FOR_EACH_29(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_28(func, arg, sep, __VA_ARGS__) sep() func(arg, 28, x)
#define MP_REVERSE_FOR_EACH_30(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_29(func, arg, sep, __VA_ARGS__) sep() func(arg, 29, x)
#define MP_REVERSE_FOR_EACH_31(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_30(func, arg, sep, __VA_ARGS__) sep() func(arg, 30, x)
#define MP_REVERSE_FOR_EACH_32(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_31(func, arg, sep, __VA_ARGS__) sep() func(arg, 31, x)
#define MP_REVERSE_FOR_EACH_33(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_32(func, arg, sep, __VA_ARGS__) sep() func(arg, 32, x)
#define MP_REVERSE_FOR_EACH_34(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_33(func, arg, sep, __VA_ARGS__) sep() func(arg, 33, x)
#define MP_REVERSE_FOR_EACH_35(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_34(func, arg, sep, __VA_ARGS__) sep() func(arg, 34, x)
#define MP_REVERSE_FOR_EACH_36(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_35(func, arg, sep, __VA_ARGS__) sep() func(arg, 35, x)
#define MP_REVERSE_FOR_EACH_37(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_36(func, arg, sep, __VA_ARGS__) sep() func(arg, 36, x)
#define MP_REVERSE_FOR_EACH_38(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_37(func, arg, sep, __VA_ARGS__) sep() func(arg, 37, x)
#define MP_REVERSE_FOR_EACH_39(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_38(func, arg, sep, __VA_ARGS__) sep() func(arg, 38, x)
#define MP_REVERSE_FOR_EACH_40(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_39(func, arg, sep, __VA_ARGS__) sep() func(arg, 39, x)
#define MP_REVERSE_FOR_EACH_41(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_40(func, arg, sep, __VA_ARGS__) sep() func(arg, 40, x)
#define MP_REVERSE_FOR_EACH_42(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_41(func, arg, sep, __VA_ARGS__) sep() func(arg, 41, x)
#define MP_REVERSE_FOR_EACH_43(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_42(func, arg, sep, __VA_ARGS__) sep() func(arg, 42, x)
#define MP_REVERSE_FOR_EACH_44(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_43(func, arg, sep, __VA_ARGS__) sep() func(arg, 43, x)
#define MP_REVERSE_FOR_EACH_45(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_44(func, arg, sep, __VA_ARGS__) sep() func(arg, 44, x)
#define MP_REVERSE_FOR_EACH_46(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_45(func, arg, sep, __VA_ARGS__) sep() func(arg, 45, x)
#define MP_REVERSE_FOR_EACH_47(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_46(func, arg, sep, __VA_ARGS__) sep() func(arg, 46, x)
#define MP_REVERSE_FOR_EACH_48(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_47(func, arg, sep, __VA_ARGS__) sep() func(arg, 47, x)
#define MP_REVERSE_FOR_EACH_49(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_48(func, arg, sep, __VA_ARGS__) sep() func(arg, 48, x)
#define MP_REVERSE_FOR_EACH_50(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_49(func, arg, sep, __VA_ARGS__) sep() func(arg, 49, x)
#define MP_REVERSE_FOR_EACH_51(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_50(func, arg, sep, __VA_ARGS__) sep() func(arg, 50, x)
#define MP_REVERSE_FOR_EACH_52(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_51(func, arg, sep, __VA_ARGS__) sep() func(arg, 51, x)
#define MP_REVERSE_FOR_EACH_53(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_52(func, arg, sep, __VA_ARGS__) sep() func(arg, 52, x)
#define MP_REVERSE_FOR_EACH_54(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_53(func, arg, sep, __VA_ARGS__) sep() func(arg, 53, x)
#define MP_REVERSE_FOR_EACH_55(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_54(func, arg, sep, __VA_ARGS__) sep() func(arg, 54, x)
#define MP_REVERSE_FOR_EACH_56(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_55(func, arg, sep, __VA_ARGS__) sep() func(arg, 55, x)
#define MP_REVERSE_FOR_EACH_57(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_56(func, arg, sep, __VA_ARGS__) sep() func(arg, 56, x)
#define MP_REVERSE_FOR_EACH_58(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_57(func, arg, sep, __VA_ARGS__) sep() func(arg, 57, x)
#define MP_REVERSE_FOR_EACH_59(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_58(func, arg, sep, __VA_ARGS__) sep() func(arg, 58, x)
#define MP_REVERSE_FOR_EACH_60(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_59(func, arg, sep, __VA_ARGS__) sep() func(arg, 59, x)
#define MP_REVERSE_FOR_EACH_61(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_60(func, arg, sep, __VA_ARGS__) sep() func(arg, 60, x)
#define MP_REVERSE_FOR_EACH_62(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_61(func, arg, sep, __VA_ARGS__) sep() func(arg, 61, x)
#define MP_REVERSE_FOR_EACH_63(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_62(func, arg, sep, __VA_ARGS__) sep() func(arg, 62, x)
#define MP_REVERSE_FOR_EACH_64(func, arg, sep, x, ...)  MP_REVERSE_FOR_EACH_63(func, arg, sep, __VA_ARGS__) sep() func(arg, 63, x)

#define MP_REVERSE_FOR_EACH_(N, func, arg, sep, ...) MP_CONCAT(MP_REVERSE_FOR_EACH_, N)(func, arg, sep, __VA_ARGS__)
#define MP_REVERSE_FOR_EACH(func, arg, sep, ...) MP_REVERSE_FOR_EACH_(MP_NARG(__VA_ARGS__), func, arg, sep, __VA_ARGS__)

#define MP_FIRST_ARG_(N, ...) N
#define MP_FIRST_ARG(...) MP_FIRST_ARG_(__VA_ARGS__, ignore)

// MP_REPEAT macro

#define MP_REPEAT_0(func, sep)
#define MP_REPEAT_1(func, sep) func(0)
#define MP_REPEAT_2(func, sep) MP_REPEAT_1(func, sep) sep func(1)
#define MP_REPEAT_3(func, sep) MP_REPEAT_2(func, sep) sep func(2)
#define MP_REPEAT_4(func, sep) MP_REPEAT_3(func, sep) sep func(3)
#define MP_REPEAT_5(func, sep) MP_REPEAT_4(func, sep) sep func(4)
#define MP_REPEAT_6(func, sep) MP_REPEAT_5(func, sep) sep func(5)
#define MP_REPEAT_7(func, sep) MP_REPEAT_6(func, sep) sep func(6)
#define MP_REPEAT_8(func, sep) MP_REPEAT_7(func, sep) sep func(7)
#define MP_REPEAT_9(func, sep) MP_REPEAT_8(func, sep) sep func(8)
#define MP_REPEAT_10(func, sep) MP_REPEAT_9(func, sep) sep func(9)
#define MP_REPEAT_11(func, sep) MP_REPEAT_10(func, sep) sep func(10)
#define MP_REPEAT_12(func, sep) MP_REPEAT_11(func, sep) sep func(11)
#define MP_REPEAT_13(func, sep) MP_REPEAT_12(func, sep) sep func(12)
#define MP_REPEAT_14(func, sep) MP_REPEAT_13(func, sep) sep func(13)
#define MP_REPEAT_15(func, sep) MP_REPEAT_14(func, sep) sep func(14)
#define MP_REPEAT_16(func, sep) MP_REPEAT_15(func, sep) sep func(15)
#define MP_REPEAT_17(func, sep) MP_REPEAT_16(func, sep) sep func(16)
#define MP_REPEAT_18(func, sep) MP_REPEAT_17(func, sep) sep func(17)
#define MP_REPEAT_19(func, sep) MP_REPEAT_18(func, sep) sep func(18)
#define MP_REPEAT_20(func, sep) MP_REPEAT_19(func, sep) sep func(19)
#define MP_REPEAT_21(func, sep) MP_REPEAT_20(func, sep) sep func(20)
#define MP_REPEAT_22(func, sep) MP_REPEAT_21(func, sep) sep func(21)
#define MP_REPEAT_23(func, sep) MP_REPEAT_22(func, sep) sep func(22)
#define MP_REPEAT_24(func, sep) MP_REPEAT_23(func, sep) sep func(23)
#define MP_REPEAT_25(func, sep) MP_REPEAT_24(func, sep) sep func(24)
#define MP_REPEAT_26(func, sep) MP_REPEAT_25(func, sep) sep func(25)
#define MP_REPEAT_27(func, sep) MP_REPEAT_26(func, sep) sep func(26)
#define MP_REPEAT_28(func, sep) MP_REPEAT_27(func, sep) sep func(27)
#define MP_REPEAT_29(func, sep) MP_REPEAT_28(func, sep) sep func(28)
#define MP_REPEAT_30(func, sep) MP_REPEAT_29(func, sep) sep func(29)
#define MP_REPEAT_31(func, sep) MP_REPEAT_30(func, sep) sep func(30)
#define MP_REPEAT_32(func, sep) MP_REPEAT_31(func, sep) sep func(31)
#define MP_REPEAT(N, func, sep) MP_CONCAT(MP_REPEAT_, N)(func, sep)
