
#include <iostream>
#include "io/cmd_parser.h"
#include "dstruct/factor_graph/factor_graph.h"
#include "app/gibbs/gibbs_sampling.h"

#ifndef _EXPMAX_H_
#define _EXPMAX_H_

namespace dd{

    /**
     * Class for (NUMA-aware) expectation maximization with gibbs sampling
     *
     * This class encapsulates expectation maximization with gibbs learning and inference, and dumping results.
     * Note the factor graph is copied on each NUMA node.
     */
    class ExpMax{
    public:
        // factor graph
        FactorGraph * const p_fg;

        // command line parser
        CmdParser * const p_cmd_parser;

        // local instance of NUMA-aware gibbs sampling
        GibbsSampling * const gibbs;


        long n_evid;
        bool * const evid_map;
        double * const old_weight_values;

        // Convergence boolean flag
        bool hasConverged;


        // convergence threshold
        double threshold;


        /**
         * Constructs ExpMax class with given factor graph, command line parser,
         * a maximum number of iterations and a delta parameter to determine convergence.
         * The number of copies, NUMA nodes and other parameters are used for
         * NUMA-aware Gibbs sampling.
         */

        ExpMax(FactorGraph * const _p_fg, CmdParser * const _p_cmd_parser, GibbsSampling * const _gibbs, double _threshold);


        /**
         * Expecation Step.
         */
        void expectation(const int & n_epoch, const bool is_quiet);


        /**
         * Maximization Step.
         */
        void maximization(const int & n_epoch, const int & n_sample_per_epoch,
                          const double & stepsize, const double & decay, const double reg_param,
                          const bool is_quiet);


        /**
         * Check convergence
         */
        void checkConvergence();

        /**
        * Preprocessing before learning. Fix a possible world
        * for all variables. Set all variables to evidence.
        */
        void sampleWorld();

        /**
         * Reset actual evidence variables.
         */
        void resetEvidence();

        /**
         * Aggregates results from different NUMA nodes
         * Dumps the inference result for variables
         * is_quiet whether to compress information display
         */
        void aggregate_results_and_dump(const bool is_quiet);

        /**
         * Dumps the learned weights
         * is_quiet whether to compress information display
         */
        void dump_weights(const bool is_quiet);

    };
}


#endif