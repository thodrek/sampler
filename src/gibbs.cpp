
#include "gibbs.h"
#include "dstruct/factor_graph/inference_result.h"

/*
 * Parse input arguments
 */
dd::CmdParser parse_input(int argc, char** argv){

  std::vector<std::string> new_args;
  if (argc < 2) {
    new_args.push_back(std::string(argv[0]) + " " + "gibbs");
    new_args.push_back("-h");
  } else {
    new_args.push_back(std::string(argv[0]) + " " + std::string(argv[1]));
    for(int i=2;i<argc;i++){
      new_args.push_back(std::string(argv[i]));
    }
  }
  char ** new_argv = new char*[new_args.size()];
  for(size_t i=0;i<new_args.size();i++){
    new_argv[i] = new char[new_args[i].length() + 1];
    std::copy(new_args[i].begin(), new_args[i].end(), new_argv[i]);
    new_argv[i][new_args[i].length()] = '\0';
  }
    std::string final_app_name;
  if (argc >= 2)
      final_app_name = argv[1];
  else
      final_app_name = "gibbs";
    
  dd::CmdParser cmd_parser(final_app_name);
  cmd_parser.parse(new_args.size(), new_argv);
  return cmd_parser;
}

void gibbs(dd::CmdParser & cmd_parser){

  // number of NUMA nodes
  int n_numa_node = numa_max_node() + 1;
  // number of max threads per NUMA node
  int n_thread_per_numa = (sysconf(_SC_NPROCESSORS_CONF))/(n_numa_node);

  // get command line arguments
  std::string fg_file = cmd_parser.fg_file->getValue();

  std::string weight_file = cmd_parser.weight_file->getValue();
  std::string variable_file = cmd_parser.variable_file->getValue();
  std::string factor_file = cmd_parser.factor_file->getValue();
  std::string edge_file = cmd_parser.edge_file->getValue();

  std::string output_folder = cmd_parser.output_folder->getValue();

  int n_learning_epoch = cmd_parser.n_learning_epoch->getValue();
  int n_samples_per_learning_epoch = cmd_parser.n_samples_per_learning_epoch->getValue();
  int n_inference_epoch = cmd_parser.n_inference_epoch->getValue();

  double stepsize = cmd_parser.stepsize->getValue();
  double stepsize2 = cmd_parser.stepsize2->getValue();
  // hack to support two parameters to specify step size
  if (stepsize == 0.01) stepsize = stepsize2;
  double decay = cmd_parser.decay->getValue();

  int n_datacopy = cmd_parser.n_datacopy->getValue();
  double reg_param = cmd_parser.reg_param->getValue();
  double reg1_param = cmd_parser.reg1_param->getValue();
  bool is_quiet = cmd_parser.quiet->getValue();
  bool sample_evidence = cmd_parser.sample_evidence->getValue();
  int burn_in = cmd_parser.burn_in->getValue();
  bool learn_non_evidence = cmd_parser.learn_non_evidence->getValue();

  Meta meta = read_meta(fg_file); 

  if (is_quiet) {
    std::cout << "Running in quiet mode..." << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "#################MACHINE CONFIG#################" << std::endl;
    std::cout << "# # NUMA Node        : " << n_numa_node << std::endl;
    std::cout << "# # Thread/NUMA Node : " << n_thread_per_numa << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << std::endl;
    std::cout << "#################GIBBS SAMPLING#################" << std::endl;
    std::cout << "# fg_file            : " << fg_file << std::endl;
    std::cout << "# edge_file          : " << edge_file << std::endl;
    std::cout << "# weight_file        : " << weight_file << std::endl;
    std::cout << "# variable_file      : " << variable_file << std::endl;
    std::cout << "# factor_file        : " << factor_file << std::endl;
    std::cout << "# output_folder      : " << output_folder << std::endl;
    std::cout << "# n_learning_epoch   : " << n_learning_epoch << std::endl;
    std::cout << "# n_samples/l. epoch : " << n_samples_per_learning_epoch << std::endl;
    std::cout << "# n_inference_epoch  : " << n_inference_epoch << std::endl;
    std::cout << "# stepsize           : " << stepsize << std::endl;
    std::cout << "# decay              : " << decay << std::endl;
    std::cout << "# regularization     : " << reg_param << std::endl;
    std::cout << "# l1 regularization     : " << reg1_param << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << "# IGNORE -s (n_samples/l. epoch). ALWAYS -s 1. #" << std::endl;
    std::cout << "# IGNORE -t (threads). ALWAYS USE ALL THREADS. #" << std::endl;
    std::cout << "################################################" << std::endl;


    std::cout << "# nvar               : " << meta.num_variables << std::endl;
    std::cout << "# nfac               : " << meta.num_factors << std::endl;
    std::cout << "# nweight            : " << meta.num_weights << std::endl;
    std::cout << "# nedge              : " << meta.num_edges << std::endl;
    std::cout << "################################################" << std::endl;
  }

  // run on NUMA node 0
  numa_run_on_node(0);
  numa_set_localalloc();

  // load factor graph
  dd::FactorGraph fg(meta.num_variables, meta.num_factors, meta.num_weights, meta.num_edges);
  fg.load(cmd_parser, is_quiet);
  dd::GibbsSampling gibbs(&fg, &cmd_parser, n_datacopy, sample_evidence, burn_in, learn_non_evidence);

  // number of learning epochs
  // the factor graph is copied on each NUMA node, so the total epochs =
  // epochs specified / number of NUMA nodes
  int numa_aware_n_learning_epoch = (int)(n_learning_epoch/n_numa_node) + 
                            (n_learning_epoch%n_numa_node==0?0:1);

  // learning
  gibbs.learn(numa_aware_n_learning_epoch, n_samples_per_learning_epoch, 
    stepsize, decay, reg_param, reg1_param, is_quiet);

  // dump weights
  gibbs.dump_weights(is_quiet);

  // number of inference epochs
  int numa_aware_n_epoch = (int)(n_inference_epoch/n_numa_node) + 
                            (n_inference_epoch%n_numa_node==0?0:1);

  // inference
  gibbs.inference(numa_aware_n_epoch, is_quiet);
  gibbs.aggregate_results_and_dump(is_quiet);


  // print weights from inference result
  for(long t=0;t<fg.n_weight;t++) {
      std::cout<<fg.infrs->weight_values[t]<<std::endl;
  }

  // print weights from factor graph
  std::cout<<"PRINTING WEIGHTS FROM WEIGHT VARIABLE"<<std::endl;
  for(long t=0;t<fg.n_weight;t++) {
      std::cout<<fg.weights[t].weight<<std::endl;
  }
}

void em(dd::CmdParser & cmd_parser){

  // number of NUMA nodes
  int n_numa_node = numa_max_node() + 1;
  // number of max threads per NUMA node
  int n_thread_per_numa = (sysconf(_SC_NPROCESSORS_CONF))/(n_numa_node);

  // get command line arguments
  std::string fg_file = cmd_parser.fg_file->getValue();

  std::string weight_file = cmd_parser.weight_file->getValue();
  std::string variable_file = cmd_parser.variable_file->getValue();
  std::string factor_file = cmd_parser.factor_file->getValue();
  std::string edge_file = cmd_parser.edge_file->getValue();

  std::string output_folder = cmd_parser.output_folder->getValue();

  int n_learning_epoch = cmd_parser.n_learning_epoch->getValue();
  int n_samples_per_learning_epoch = cmd_parser.n_samples_per_learning_epoch->getValue();
  int n_inference_epoch = cmd_parser.n_inference_epoch->getValue();

  double stepsize = cmd_parser.stepsize->getValue();
  double stepsize2 = cmd_parser.stepsize2->getValue();
  // hack to support two parameters to specify step size
  if (stepsize == 0.01) stepsize = stepsize2;
  double decay = cmd_parser.decay->getValue();

  int n_datacopy = cmd_parser.n_datacopy->getValue();
  double reg_param = cmd_parser.reg_param->getValue();
  double reg1_param = cmd_parser.reg1_param->getValue();
  bool is_quiet = cmd_parser.quiet->getValue();
  bool check_convergence = cmd_parser.check_convergence->getValue();
  bool sample_evidence = cmd_parser.sample_evidence->getValue();
  int burn_in = cmd_parser.burn_in->getValue();
  int n_iter = cmd_parser.n_iter->getValue();
  int wl_conv = cmd_parser.wl_conv->getValue();
  int delta = cmd_parser.delta->getValue();
  bool learn_non_evidence = cmd_parser.learn_non_evidence->getValue();

  Meta meta = read_meta(fg_file);

  if (is_quiet) {
    std::cout << "Running in quiet mode..." << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "#################MACHINE CONFIG#################" << std::endl;
    std::cout << "# # NUMA Node        : " << n_numa_node << std::endl;
    std::cout << "# # Thread/NUMA Node : " << n_thread_per_numa << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << std::endl;
    std::cout << "#################GIBBS SAMPLING#################" << std::endl;
    std::cout << "# fg_file            : " << fg_file << std::endl;
    std::cout << "# edge_file          : " << edge_file << std::endl;
    std::cout << "# weight_file        : " << weight_file << std::endl;
    std::cout << "# variable_file      : " << variable_file << std::endl;
    std::cout << "# factor_file        : " << factor_file << std::endl;
    std::cout << "# output_folder      : " << output_folder << std::endl;
    std::cout << "# n_learning_epoch   : " << n_learning_epoch << std::endl;
    std::cout << "# n_samples/l. epoch : " << n_samples_per_learning_epoch << std::endl;
    std::cout << "# n_inference_epoch  : " << n_inference_epoch << std::endl;
    std::cout << "# stepsize           : " << stepsize << std::endl;
    std::cout << "# decay              : " << decay << std::endl;
    std::cout << "# regularization     : " << reg_param << std::endl;
    std::cout << "# l1 regularization     : " << reg1_param << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << "# IGNORE -s (n_samples/l. epoch). ALWAYS -s 1. #" << std::endl;
    std::cout << "# IGNORE -t (threads). ALWAYS USE ALL THREADS. #" << std::endl;
    std::cout << "################################################" << std::endl;


    std::cout << "# nvar               : " << meta.num_variables << std::endl;
    std::cout << "# nfac               : " << meta.num_factors << std::endl;
    std::cout << "# nweight            : " << meta.num_weights << std::endl;
    std::cout << "# nedge              : " << meta.num_edges << std::endl;
    std::cout << "################################################" << std::endl;
  }

  // run on NUMA node 0
  numa_run_on_node(0);
  numa_set_localalloc();

  // load factor graph
  dd::FactorGraph fg(meta.num_variables, meta.num_factors, meta.num_weights, meta.num_edges);
  fg.load(cmd_parser, is_quiet);
  dd::GibbsSampling gibbs(&fg, &cmd_parser, n_datacopy, sample_evidence, burn_in, learn_non_evidence);

  // Initialize EM instance
  dd::ExpMax expMax(&fg, &gibbs, wl_conv, delta, check_convergence);

  // number of inference epochs
  int numa_aware_n_epoch;
  int numa_aware_n_learning_epoch;

  // EM init -- run Maximzation (semi-supervised learning)

  // Maximization step
  numa_aware_n_learning_epoch = (int)(n_learning_epoch/n_numa_node) +
                                (n_learning_epoch%n_numa_node==0?0:1);
  expMax.maximization(numa_aware_n_learning_epoch, n_samples_per_learning_epoch,
                      stepsize, decay, reg_param, reg1_param, is_quiet);

  while (!expMax.hasConverged && n_iter > 0) {

    // Expectation step
    numa_aware_n_epoch = (int)(n_inference_epoch/n_numa_node) +
                         (n_inference_epoch%n_numa_node==0?0:1);
    expMax.expectation(numa_aware_n_epoch,is_quiet);


    // Maximization step
    numa_aware_n_learning_epoch = (int)(n_learning_epoch/n_numa_node) +
                                  (n_learning_epoch%n_numa_node==0?0:1);
    expMax.maximization(numa_aware_n_learning_epoch, n_samples_per_learning_epoch,
                        stepsize, decay, reg_param, reg1_param, is_quiet);

    //Decrement iteration counter
    n_iter--;
  }

  expMax.dump_weights(is_quiet);
  expMax.aggregate_results_and_dump(is_quiet);


}
