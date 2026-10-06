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
REG rd[2];

REG q2_rd[2];
bool q2_isLoad[2] = {false, false};

int insCount    = 0; //number of dynamically executed instructions
int loads       = 0;
int lduh        = 0;
int stalls_q1   = 0;
int stalls_q2   = 0;

int single_stalls = 0;
int double_stalls = 0;

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

  int current_stalls_q2 = 0;

  for (int i = 0; (size_t)i < srcs->size(); i++) {
    if (i != storeValueOp && reg_ready[(*srcs)[i]] > insCount) {
      lduh++;
      break;
    }

    // Q1 (5-stage, no forwarding)
    if((*srcs)[i] == rd[0]) {
      stalls_q1 += 2;
      double_stalls++;
      rd[0] = REG_INVALID_;
    }

    if((*srcs)[i] == rd[1]) {
      stalls_q1 += 1;
      single_stalls++;
      rd[1] = REG_INVALID_;
    }

    //Q2 (6-stage, full forwarding)
    // dist 1 (immediate predecessor)
    if((*srcs)[i] == q2_rd[0]) {
      if (q2_isLoad[0]) {
          current_stalls_q2 = std::max(current_stalls_q2, 2); // load -> 2 stalls
      } else {
          current_stalls_q2 = std::max(current_stalls_q2, 1); // ALU -> 1 stall
      }
      q2_rd[0] = REG_INVALID_; // Clear to prevent double counting
    }
    
    // dist 2 (separated by 1 instruction)
    if((*srcs)[i] == q2_rd[1]) {
      if (q2_isLoad[1]) {
          current_stalls_q2 = std::max(current_stalls_q2, 1); // load -> 1 stall
      } 
      // ALU at distance 2 is 0 stalls, so we do nothing
      q2_rd[1] = REG_INVALID_;
    }
  }

  stalls_q2 += current_stalls_q2;

  rd[1] = rd[0];
  rd[0] = REG_INVALID_;

  q2_rd[1] = q2_rd[0];
  q2_isLoad[1] = q2_isLoad[0];
  q2_rd[0] = REG_INVALID_;
  q2_isLoad[0] = false;

  for (REG dst : *dsts) {
    if (isLoad) {
      reg_ready[dst] = insCount + 2;
    }

    rd[0] = dst;
    q2_rd[0] = dst;
    q2_isLoad[0] = isLoad;
  }


}
/* ===================================================================== */
// Instrumentation callbacks
/* ===================================================================== */

VOID Instruction(INS insn, VOID* v)
{
  std::vector<REG> *srcs = new std::vector<REG>();
  std::vector<REG> *dsts = new std::vector<REG>();
  for (UINT32 i = 0; i < INS_OperandCount(insn); i++)
  {
    //For simplicity, only allow 1 of these options (in reality multiple are possible in x86)
    REG op = INS_OperandReg(insn, i);
    if (op == REG_INVALID())
    {
      op = INS_OperandMemoryBaseReg(insn, i);
    }
    if (op == REG_INVALID())
    {
      op = INS_OperandMemoryIndexReg(insn, i);
    }
    if (op != REG_INVALID())
    {
      if (INS_OperandRead(insn, i) || INS_OperandIsMemory(insn, i)) srcs->push_back(op);
      if (INS_OperandWritten(insn, i) && !INS_OperandIsMemory(insn, i)) dsts->push_back(op);
    }
  }
  INS_InsertCall(insn, IPOINT_BEFORE, (AFUNPTR)CountLDUH,
      IARG_ADDRINT, srcs,
      IARG_ADDRINT, dsts,
      IARG_BOOL, INS_IsMemoryRead(insn),
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
    cerr << "===============================================" << endl;

    // Q1
    fprintf(stderr, "Question 1 stall cycles: %d CPI %lf slowdown: %lf%%\n",
            stalls_q1,
            1.0 + (float) stalls_q1 / insCount,
            (float)(float(stalls_q1 / (float(insCount + stalls_q1))) * 100.0));
    cerr << "===============================================" << endl;
    // Q2
    fprintf(stderr, "Question 2 stall cycles: %d CPI %lf slowdown: %lf%%\n",
            stalls_q2,
            1.0 + (float) stalls_q2 / insCount,
            (float)(float(stalls_q2 / (float(insCount + stalls_q2))) * 100.0));
    cerr << "===============================================" << endl;
    fprintf(stderr, "Single Stalls: %d\nDouble Stalls: %d\n", single_stalls, double_stalls);
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
