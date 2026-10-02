/*
 * Copyright (C) 2007-2021 Intel Corporation.
 * SPDX-License-Identifier: MIT
 */

/*! @file
 *  This is an example of the PIN tool that demonstrates some basic PIN APIs 
 *  and could serve as the starting point for developing your first PIN tool
 */

#include "pin.H"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
using std::cerr;
using std::endl;
using std::string;

/* ================================================================== */
// Global variables
/* ================================================================== */

int reg_ready[REG_LAST];

int insCount    = 0; //number of dynamically executed instructions
int loads       = 0;
int lduh        = 0;
int stalls_q1   = 0;
int stalls_q2   = 0;

/* ===================================================================== */
// Command line switches
/* ===================================================================== */

/* ===================================================================== */
// Utilities
/* ===================================================================== */
//Returns -1 if the instruction is not a simple store
//Otherwise, returns the index of the operand that provides the store value
int storeOps(INS insn)
{
  if (!INS_IsMemoryWrite(insn)) return -1;
  if (INS_IsMov(insn)) return 1;
  if (!INS_IsXchg(insn)) return -1;
  if (INS_OperandIsMemory(insn, 0)) return 1;
  return 0;
}
/* ===================================================================== */
// Analysis routines
/* ===================================================================== */

void CountLDUH(
std::vector<REG>* srcs,
std::vector<REG>* dsts,
bool isLoad,
int storeValueOp)
{
  insCount++;
  if (isLoad) {
    loads++;
  }
  for (REG dst : *dsts) {
    if (isLoad) {
      reg_ready[dst] = insCount + 2;
    }
  }

  for (int i = 0; (size_t)i < srcs->size(); i++) {
    if (i != storeValueOp && reg_ready[(*srcs)[i]] > insCount) {
      lduh++;
      break;
    }
  }
}
/* ===================================================================== */
// Instrumentation callbacks
/* ===================================================================== */


VOID Instruction(INS insn, VOID* v)
{
  std::vector<REG>* srcs = new std::vector<REG>();
  std::vector<REG>* dsts = new std::vector<REG>();
  bool isLoad = INS_IsMemoryRead(insn);

  for (UINT32 i = 0; i < INS_OperandCount(insn); i++)
  {
    REG reg = INS_OperandReg(insn, i);
    if (reg == REG_INVALID()) continue;
    if (INS_OperandRead(insn, i)) srcs->push_back(reg);
    if (INS_OperandWritten(insn, i)) dsts->push_back(reg);
  }

  INS_InsertCall(insn, IPOINT_BEFORE, (AFUNPTR)CountLDUH, 
    IARG_ADDRINT, srcs,
    IARG_ADDRINT, dsts,
    IARG_BOOL, isLoad,
      IARG_ADDRINT, storeOps(insn),
      IARG_END);
}

/*!
 * Print out analysis results.
 * This function is called when the application exits.
 * @param[in]   code            exit code of the application
 * @param[in]   v               value specified by the tool in the 
 *                              PIN_AddFiniFunction function call
 */
VOID Fini(INT32 code, VOID* v)
{
    cerr << "===============================================" << endl;
    cerr << "Load-to-use Hazard analysis results: " << endl;
    cerr << "Number of instructions: " << insCount << endl;
    cerr << "Number of load instructions: " << loads << endl;
    cerr << "Number of load-to-use hazards: " << lduh << endl;
    fprintf(stderr, "Question 1 stall cycles: %d CPI %lf slowdown: %lf%%\n", stalls_q1, 0.0, 0.0);
    fprintf(stderr, "Question 2 stall cycles: %d CPI %lf slowdown: %lf%%\n", stalls_q2, 0.0, 0.0);
    cerr << "===============================================" << endl;
}

/*!
 * The main procedure of the tool.
 * This function is called when the application image is loaded but not yet started.
 * @param[in]   argc            total number of elements in the argv array
 * @param[in]   argv            array of command line arguments, 
 *                              including pin -t <toolname> -- ...
 */
int main(int argc, char* argv[])
{
    // Initialize PIN library. Print help message if -h(elp) is specified
    // in the command line or the command line is invalid
    if (PIN_Init(argc, argv))
    {
        return -1;
    }

    INS_AddInstrumentFunction(Instruction, 0);
    // Register function to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);


    // Start the program, never returns
    PIN_StartProgram();

    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
