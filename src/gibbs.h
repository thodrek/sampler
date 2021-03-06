#include <assert.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include "common.h"
#include <unistd.h>

#include "io/cmd_parser.h"
#include "io/binary_parser.h"

#include "app/gibbs/gibbs_sampling.h"
#include "app/em/expmax.h"
#include "dstruct/factor_graph/factor_graph.h"

/*
 * Parse input arguments
 */
dd::CmdParser parse_input(int argc, char** argv);

/**
 * Runs gibbs sampling using the given command line parser
 */
void gibbs(dd::CmdParser & cmd_parser);

/**
 * Runs expecation maximization using the underlying gibbs sampler and the given command line parser
 */
void em(dd::CmdParser & cmd_parser);







