#!/Library/Frameworks/Python.framework/Versions/3.4/bin/python3

import os
import argparse
import csv
import datetime
import logging
import random
import sys
import time
import numpy as np
import helper

global version

if __name__ == "__main__":
    global version
    version = "0.0.5"
    run_id = random.randint(1, 100000)
    now = datetime.datetime.now()

    args = None
    parser = argparse.ArgumentParser(description='Deep Learning Assisted Heuristic Tree Search.')
    parser.add_argument('-l', '--labeled_data_dir', type=str, default="",
                        help='Labeled training data directory. WARNING previously named labeld_data_dir')
    parser.add_argument('-t', '--test_data_dir', type=str, default="", help='Labeled test data directory.')
    parser.add_argument('-m', '--model_path', type=str, default="", help='Pre-learned branching DNN model path.')
    parser.add_argument('-v', '--value_model_path', type=str, default="", help='Pre-learned value DNN model path.')
    parser.add_argument('-o', '--output_path', type=str, default="", help='Output path.')
    parser.add_argument('-s', '--use_value_model', action='store_true', help='Indicates whether or not to train/use a value model during search.')
    parser.add_argument('-b', '--verbose', type=int, default=1, help='Verbosity level. 0 = quiet, 1 = normal, 2 = debug')
    parser.add_argument('--max_n', type=int, default=10**10, help='Maximum number of training examples for the policy/value networks.')
    parser.add_argument('--ub', type=int, default=45, help='Maximum depth of the tree search until an valid solution is found.')
    valid_branch_strategies = ["constant", "linear", "quadratic", "log"]
    parser.add_argument('--branch_strategy', type=str, default="log", help="Branching strategy. Options: {{{0}}}".format(", ".join(valid_branch_strategies)))
    parser.add_argument('-p', '--tree_search_width', type=float, default=0.1, help='Tree search width (parameter p)')
    parser.add_argument('-n', type=int, default=3, help='Frequency in which the value DNN is queried')
    parser.add_argument('--val_timeout', type=float, default=60.0, help='Maximum CPU time per validation run')
    parser.add_argument('--param_d', type=float, default=0.95,
                        help='Value DNN reduction factor (to try to avoid overestimation)')
    parser.add_argument('--param_d2', type=float, default=1.0,
                        help='Value DNN increase/decrease factor for the WBS')
    parser.add_argument('--param_e', type=float, default=0.95,
                        help='f = g * e + h; reduction factor e for the path already explored.')
    parser.add_argument('--p_a_1', nargs='+', type=int, default=[6, 4, 3],
                        help='Architecture of the layers with shared weights of the policy network.'
                             'For example "3 2" results in 2 layers with tiers*3 units'
                             'and tiers*stacks*2 units respectively')
    parser.add_argument('--p_a_2', nargs='+', type=int, default=[9, 6, 2],
                        help='Architecture of the layers without shared weights of the policy network.'
                             'For example "3 2" results in 2 layers with tiers*stacks*3 units'
                             'and tiers*stacks*2 units respectively')
    parser.add_argument('--v_a_1', nargs='+', type=int, default=[4, 3, 2],
                        help='Architecture of the layers with shared weights of the value network.'
                             'For example "3 2" results in 2 layers with tiers*3 units'
                             'and tiers*stacks*2 units respectively')
    parser.add_argument('--v_a_2', nargs='+', type=int, default=[3, 2, 2],
                        help='Architecture of the layers without shared weights of the value network.'
                             'For example "3 2" results in 2 layers with tiers*stacks*3 units'
                             'and tiers*stacks*2 units respectively')
    parser.add_argument('--p_b', type=int, default=512,
                        help='Batch size used for the training of the policy network')
    parser.add_argument('--v_b', type=int, default=512,
                        help='Batch size used for the training of the value network')
    parser.add_argument('--p_l', type=float, default=0.001,
                        help='Learning rate for the policy network')
    parser.add_argument('--v_l', type=float, default=0.001,
                        help='Learning rate for the the value network')
    valid_search_strategies = ["dfs", "lds", "wbs"]
    parser.add_argument('--search_strategy', type=str, default="dfs", help="Tree exploration strategy. Options: {{{0}}}".format(valid_search_strategies))
    parser.add_argument('--lds_use_bins', action='store_true', help='Use a binning strategy for the LDS.')
    parser.add_argument('--lds_bins', type=int, default=5, help='Number of bins for the LDS bin strategy.')
    parser.add_argument('--lds_zero_depth', type=int, default=-1, help='Depth up to which the LDS search strategy should set the discrepency of all branches to 0.')
    parser.add_argument('--tuning', action='store_true', help='Set this flag when tuning with GGA or some other algorithm configurator.')
    parser.add_argument('--val_instances', type=str, default=None, help='Selects which validation instances to run in a comma separated list. Leave this parameter out to run all of them.')
    parser.add_argument('--training_seed', type=int, default=0, help='Seed for the random assignment of instances to the training or validation set.')
    parser.add_argument('--run_id', type=int, default=-1, help='Run id; for use only in tuning situations to avoid issues with theano\'s compile lock')
    parser.add_argument('--dummy_tiers', type=int, default=0, help='Number of dummy tiers that should be added below the tiers of an instance to create compatiblity with DNNs for lager instances.')

    args = parser.parse_args()

    model_path = args.model_path
    value_model_path = args.value_model_path
    output_path = args.output_path
    test_data_dir = args.test_data_dir
    labeled_data_dir = args.labeled_data_dir
    use_value_model = args.use_value_model
    param_ub = args.ub
    param_p = args.tree_search_width
    param_n = args.n
    param_d = args.param_d
    param_d2 = args.param_d2
    param_e = args.param_e
    param_max_n = args.max_n
    param_v_a_1 = args.v_a_1
    param_v_a_2 = args.v_a_2
    param_p_a_1 = args.p_a_1
    param_p_a_2 = args.p_a_2
    param_p_b = args.p_b
    param_v_b = args.v_b
    param_p_l = args.p_l
    param_v_l = args.v_l
    verbosity = args.verbose
    vtimeout = args.val_timeout - 0.1
    bstrategy = args.branch_strategy
    sstrategy = args.search_strategy
    use_lds_bins = args.lds_use_bins
    lds_bins = args.lds_bins
    lds_zero_depth = args.lds_zero_depth
    tuning = args.tuning
    val_insts = args.val_instances
    training_seed = args.training_seed
    dummy_tiers = args.dummy_tiers
    if args.run_id > 0:
        run_id = args.run_id

    if val_insts:
        val_insts = [int(vv) for vv in val_insts.split(',')]

    if args.tuning:
        print(" ".join(sys.argv), file=sys.stderr)

    # TODO more helpful output when an incorrect branching/search strategy is given
    if (not model_path and not labeled_data_dir) or bstrategy not in valid_branch_strategies or sstrategy not in valid_search_strategies:
        parser.print_help()
        sys.exit(1)
    if sstrategy == 'wbs' and not use_value_model:
        print("BFS requires the use of a value network.")
        sys.exit(1)

    #os.environ['THEANO_FLAGS'] = "base_compiledir=/tmp/kbt/theano/{0}/".format(run_id)
    import instance_solver
    import policy_network
    import value_network
    if output_path == "":
        output_path = os.path.join(os.getcwd(), "output", "run_" + str(now.day) + "." + str(now.month) +
                                   "." + str(now.year) + "_" + str(run_id))
    if not os.path.exists(os.path.join(output_path, "models")):
        os.makedirs(os.path.join(output_path, "models"))
    if not os.path.exists(os.path.join(output_path, "solutions")):
        os.makedirs(os.path.join(output_path, "solutions"))

#     #if tuning:
#     #    logging.basicConfig(stream=os.devnull)
#     #    verbosity = -1
#     #else:
    if tuning:
        verbosity = -1
    logging.basicConfig(
        filename=os.path.join(output_path, "log_" + version + "_" + str(now.day) + "." + str(now.month) +
                              "." + str(now.year) + "_" + str(run_id) + ".log"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Started")

    from keras.models import load_model

    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(args)):
        logging.info("{0}: {1}".format(arg, getattr(args, arg)))
    logging.info("----------")

    start_time_training = time.process_time()
    if model_path == "":
        logging.info("Start training policy model")
        stacks, tiers, res, labels, labels_v, ref_rt = helper.parse_dir(labeled_data_dir, param_max_n, training_seed,
                                                                        dummy_tiers)
        branch_network = policy_network.learn(np.array(res), np.array(labels), stacks, tiers, run_id, output_path,
                                              param_p_a_1, param_p_a_2, param_p_b, param_p_l)
    else:
        branch_network = load_model(model_path)
        logging.info("Loaded policy model")

    if use_value_model:
        if value_model_path == "":
            logging.info("Start training value model")
            if model_path != "":
                stacks, tiers, res, labels, labels_v, ref_rt = helper.parse_dir(labeled_data_dir, param_max_n,
                                                                                training_seed, dummy_tiers)
            value_model = value_network.learn(np.array(res), np.array(labels_v), stacks, tiers, run_id, output_path,
                                              param_v_a_1, param_v_a_2, param_v_b, param_v_l)
        else:
            logging.info("Loaded value model")
            value_model = load_model(value_model_path)

    logging.info("Finished Training. Needed runtime: {0}".format(time.process_time() - start_time_training))
    logging.info("Start solving test instances")
    start_time_test = time.process_time()
    if not use_value_model:
        value_model = None
    res = instance_solver.solve_instances(branch_network, test_data_dir, output_path, param_ub,param_p,
                                          param_d, param_d2, param_n, dummy_tiers, verbosity, value_model, vtimeout,
                                          bstrategy, sstrategy, use_lds_bins, lds_bins,
                                          lds_zero_depth, param_e, tuning, val_insts)
    (solved_count, unsolved_count, moves_sum,ref_moves_sum, rt_sum, ref_rt_sum, results) = res

    with open(os.path.join(output_path,
                           "results_" + str(run_id) + "_" + test_data_dir.split('/')[-1] + ".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "id", "moves", "time", "opt", "nodes"])
        writer.writerows(results)

    logging.info("FINAL RESULTS:")
    logging.info("Instances solved: {0}/{1}".format(solved_count,(solved_count+unsolved_count)))
    logging.info("Total move count (Reference Move Count): {0} ({1})".format(moves_sum,ref_moves_sum))
    if solved_count > 0:
        moves_avg = round(moves_sum / solved_count, 2)
        ref_moves_avg = round(ref_moves_sum / solved_count, 2)
        rt_avg = round(rt_sum / solved_count, 2)
        ref_rt_avg = round(ref_rt_sum / (unsolved_count+ solved_count), 2)
        gap = round((moves_sum/ref_moves_sum-1)*100,2)
        logging.info("Gap to reference solutions in %: {0}".format(gap))
        logging.info("Mean Move Count (Reference Move Count): {0} ({1})".format(moves_avg, ref_moves_avg))
        logging.info("Mean Runtime (Reference Runtime): {0} ({1})".format(rt_avg, ref_rt_avg))

    if verbosity >= 0:
        print("Runtime (CPU) All: {0}".format(time.process_time() - start_time_test))
    if tuning:
        print("{0}".format(time.process_time() - start_time_test))

