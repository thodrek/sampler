
#include <iostream>
#include "io/cmd_parser.h"
#include "dstruct/factor_graph/factor_graph.h"
#include "app/gibbs/gibbs_sampling.h"
#include "common.h"

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


        // local instance of NUMA-aware gibbs sampling
        GibbsSampling * const gibbs;


        long n_evid;
        bool * const evid_map;
        double * const old_weight_values;

        // Convergence variables
        bool hasConverged;
        // convergence threshold = 10^-delta
        int delta;
        // ps-ll moving overage window
        int wl_conv;
        std::vector<double> neg_psll_buff;
        int iterationCount;

        bool check_convergence;



        /**
         * Constructs ExpMax class with given factor graph, command line parser,
         * a maximum number of iterations and a delta parameter to determine convergence.
         * The number of copies, NUMA nodes and other parameters are used for
         * NUMA-aware Gibbs sampling.
         */

        ExpMax(FactorGraph * const _p_fg, GibbsSampling * const _gibbs, int _wl_conv, int _delta, bool check_convergence);


        /**
         * Expecation Step.
         */
        void expectation(const int & n_epoch, const bool is_quiet);


        /**
         * Maximization Step.
         */
        void maximization(const int & n_epoch, const int & n_sample_per_epoch,
                          const double & stepsize, const double & decay, const double reg_param,
                          const double reg1_param, const std::string meta_file, const bool is_quiet);

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


        /**
         * Compute the negatibe pseudo-loglikelihood of observed variables
         */
        double neg_ps_loglikelihood();


        /**
         * Update the negative pseudo-loglikelihood buffer
         */
        void update_psll_buff(double npsll);


        /**
         * Check convergence of EM based on past pseudo-likelihood values
         */
        void checkConvergence();


    };
}


#endif
