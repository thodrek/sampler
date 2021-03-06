#include "app/em/expmax.h"
#include "dstruct/factor_graph/variable.h"
#include "dstruct/factor_graph/weight.h"
#include "dstruct/factor_graph/inference_result.h"
#include "common.h"


dd::ExpMax::ExpMax(FactorGraph * const _p_fg, GibbsSampling * const _gibbs, int _wl_conv, int _delta, bool _check_convergence)
: p_fg(_p_fg), gibbs(_gibbs), evid_map(new bool[_p_fg->n_var]), old_weight_values(new double[_p_fg->n_weight]), delta(_delta), wl_conv(_wl_conv), check_convergence(_check_convergence) {

    //Init iteration count
    iterationCount = 0;

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
    double neg_ps_ll = neg_ps_loglikelihood();
    std::cout<<"Neg. PSLL = "<<neg_ps_ll<<std::endl;
    if (check_convergence)
        update_psll_buff(neg_ps_ll);
}


void dd::ExpMax::maximization(const int &n_epoch, const int &n_sample_per_epoch, const double &stepsize,
const double &decay, const double reg_param, const double reg1_param, const bool is_quiet) {
//const double &decay, const double reg_param, const double reg1_param, const std::string meta_file, const bool is_quiet) {
    //this->gibbs->learn(n_epoch, n_sample_per_epoch, stepsize,decay, reg_param, reg1_param, meta_file, is_quiet);
    this->gibbs->learn(n_epoch, n_sample_per_epoch, stepsize,decay, reg_param, reg1_param, is_quiet);
    resetEvidence();
    iterationCount++;
    if (check_convergence)
        checkConvergence();
}

void dd::ExpMax::checkConvergence() {
    if (iterationCount < 2*wl_conv)
        this->hasConverged = false;
    else {
        //compute the two sliding window averages
        double oldAvg = 0.0;
        double newAvg = 0.0;

        for (std::vector<double>::iterator it = neg_psll_buff.begin() ; it != neg_psll_buff.begin()+wl_conv; ++it) {
            oldAvg+= *it;
        }
        oldAvg += oldAvg/wl_conv;

        for (std::vector<double>::iterator it = neg_psll_buff.begin()+wl_conv ; it != neg_psll_buff.end(); ++it) {
            newAvg+= *it;
        }
        newAvg += newAvg/wl_conv;
        std::cout<<"Old avg = "<<oldAvg<<std::endl;
        std::cout<<"New avg = "<<newAvg<<std::endl;
        //check convergence
        if (fabs(oldAvg - newAvg)/oldAvg <= pow (10.0, -1*delta))
            this->hasConverged = true;
        else
            this->hasConverged = false;
    }

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

double dd::ExpMax::neg_ps_loglikelihood() {
    double potential_pos;
    double potential_neg;
    double obs_inv_cond_prob;

    // these are used for calculating potentials and probabilities
    double denom_sum;

    double neg_ps_ll = 0.0;
    for (long t=0; t < this->p_fg->n_var; t++) {
        Variable & variable = this->p_fg->variables[t];
        if (variable.is_evid) {
            if(variable.domain_type == DTYPE_BOOLEAN) {

                //compute conditional probability of variable
                potential_pos = p_fg->potential<false>(variable, 1);
                potential_neg = p_fg->potential<false>(variable, 0);

                if (p_fg->infrs->assignments_evid[t] == 1)
                    obs_inv_cond_prob = 1.0 + exp(potential_neg - potential_pos);
                else
                    obs_inv_cond_prob = 1.0 + exp(potential_pos - potential_neg);

                neg_ps_ll += log(obs_inv_cond_prob);
            }
            else if(variable.domain_type == DTYPE_MULTINOMIAL){
                for(int propose=variable.lower_bound;propose <= variable.upper_bound; propose++){
                    denom_sum = exp(p_fg->potential<false>(variable, propose));
                }
                obs_inv_cond_prob = denom_sum/exp(p_fg->potential<false>(variable, p_fg->infrs->assignments_evid[t]));
                neg_ps_ll += log(obs_inv_cond_prob);

            }else{
                std::cerr << "[ERROR] Only Boolean and Multinomial variables are supported now!" << std::endl;
                assert(false);
                return -1;
            }
        }
    }
    return neg_ps_ll;
};

void dd::ExpMax::update_psll_buff(double npsll) {
    //erase fist element if needed
    if (iterationCount >= 2*wl_conv)
        neg_psll_buff.erase(neg_psll_buff.begin());
    //add new element
    neg_psll_buff.push_back(npsll);
}

