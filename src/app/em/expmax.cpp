#include "app/em/expmax.h"
#include "dstruct/factor_graph/variable.h"
#include "dstruct/factor_graph/weight.h"
#include "dstruct/factor_graph/inference_result.h"
#include "common.h"
#include "expmax.h"
#include "../../dstruct/factor_graph/variable.h"
#include "../../dstruct/factor_graph/factor_graph.h"
#include "../../dstruct/factor_graph/inference_result.h"


dd::ExpMax::ExpMax(FactorGraph * const _p_fg, CmdParser * const _p_cmd_parser, GibbsSampling * const _gibbs, double _threshold)
: p_fg(_p_fg), p_cmd_parser(_p_cmd_parser), gibbs(_gibbs), evid_map(new bool[_p_fg->n_var]), old_weight_values(new double[_p_fg->n_weight]),threshold(_threshold) {

    //Store ids of evidence variables
    for(long t=0;t<p_fg->n_var;t++) {
        const Variable &variable = this->p_fg->variables[t];
        if (variable.is_evid)
            evid_map[t] = true;
        else
            evid_map[t] = false;

    }

    //Store initial weights
    for(long t=0;t<p_fg->n_weight;t++){
        const Weight & weight = p_fg->weights[t];
        old_weight_values[weight.id] = weight.weight;
    }

    //Initialize convergence flag
    hasConverged = false;
}



void dd::ExpMax::expectation(const int &n_epoch, const bool is_quiet) {
    //perform inference using the sampler
    this->gibbs->inference(n_epoch,is_quiet);
    sampleWorld();
    //compute negative pseudo-likelihood of observed variables
    neg_ps_loglikelihood(is_quiet);
    std::cout<<"Neg. PSLL = "<<neg_ps_ll<<std::endl;
}


void dd::ExpMax::maximization(const int &n_epoch, const int &n_sample_per_epoch, const double &stepsize,
const double &decay, const double reg_param, const bool is_quiet) {
    this->gibbs->learn(n_epoch, n_sample_per_epoch, stepsize,decay, reg_param, is_quiet);
    resetEvidence();
    checkConvergence();
}

void dd::ExpMax::checkConvergence() {
    double maxdiff = 0.0;
    double tmpdiff = 0.0;
    for(long t=0;t<p_fg->n_weight;t++){
        tmpdiff = fabs(p_fg->infrs->weight_values[t] - old_weight_values[t]);
        maxdiff = tmpdiff > maxdiff ? tmpdiff : maxdiff;
        old_weight_values[t] = p_fg->infrs->weight_values[t];
    }
    if (maxdiff < this->threshold)
        this->hasConverged = true;
    else
        this->hasConverged = false;
    std::cout<<"Difference = "<<maxdiff<<std::endl;
}


void dd::ExpMax::sampleWorld() {
    for(long t=0;t<p_fg->n_var;t++) {
        p_fg->infrs->assignments_evid[t] = p_fg->infrs->assignments_free[t];
        this->p_fg->variables[t].is_evid = true;
    }
}

void dd::ExpMax::resetEvidence() {
    for(long t=0;t<p_fg->n_var;t++) {
        this->p_fg->variables[t].is_evid = evid_map[t];
    }
}

void dd::ExpMax::aggregate_results_and_dump(const bool is_quiet) {
    this->gibbs->aggregate_results_and_dump(is_quiet);
}

void dd::ExpMax::dump_weights(const bool is_quiet) {
    this->gibbs->dump_weights(is_quiet);
}

void dd::ExpMax::neg_ps_loglikelihood(const bool is_quiet) {
    double potential_pos;
    double potential_neg;
    double obs_inv_cond_prob;
    neg_ps_ll = 0.0;
    for (long t=0; this->p_fg->n_var; t++) {
        Variable & variable = this->p_fg->variables[t];
        if (variable.is_evid) {
            if(variable.domain_type == DTYPE_BOOLEAN){
                std::cout<<"Boolean"<<std::endl;
            }
            else if (variable.domain_type == DTYPE_REAL){
                std::cout<<"Real"<<std::endl;
            }
            else {
                std::cout<<"Multinomial"<<std::endl;
            }
            if(variable.domain_type == DTYPE_BOOLEAN){

                //compute conditional probability of variable
                potential_pos = p_fg->template potential<false>(variable, 1);
                potential_neg = p_fg->template potential<false>(variable, 0);

                if(p_fg->infrs->assignments_evid[t] == 1)
                    obs_inv_cond_prob = 1.0 + exp(potential_neg - potential_pos);
                else
                    obs_inv_cond_prob = 1.0 + exp(potential_pos - potential_neg);

                neg_ps_ll += log(obs_inv_cond_prob);
            }else{
                std::cerr << "[ERROR] Only Boolean variables are supported now!" << std::endl;
                assert(false);
                return;
            }
        }
    }
};

