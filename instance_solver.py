import os
import helper
import tree_search
import logging
from solution_cpmp import solution_cpmp
import sys
import glob

# TODO in some places the logger is used, here we have print. Consistency?

def solve_instances(branch_network, ddir, output_path, upper_bound, ts_param_p, ts_param_d, ts_param_d2, ts_param_n,
                    dummy_tiers, verbose=1, value_model=None, val_timeout=60.0, bstrategy='log', sstrategy='dfs',
                    use_bins=False, nbins=5, zero_depth=-1, ts_param_e=0.95, tuning=False, val_insts=None):
    solved = 0
    unsolved = 0
    ref_moves_sum = 0
    ref_rt_sum = 0
    moves_sum = 0
    rt_sum = 0
    results = []

    if val_insts:
        insts = ["{0}/{1}.out".format(ddir, vi) for vi in val_insts]
    else:
        insts = glob.glob("{0}/*.out".format(ddir))

    for ff in sorted(insts):
        # check if the instance is one we want to solve
        ff = os.path.basename(ff)
        inst_joined = os.path.join(ddir, ff)
        if verbose >= 2:
            print("Solving instance {0}".format(inst_joined))
        with open(inst_joined, 'r') as fp:
            stacks, tiers, rr, ll, ll_v, ref_rt = helper.parse_file_pointer(fp, dummy_tiers)
        if len(rr) < 1:
            continue
        cells = rr[0]
        solution = solution_cpmp(cells.copy(), stacks, tiers, dummy_tiers, branch_network, value_model)
#         start_time_test = time.time() # use .process_time
        if sstrategy == 'dfs':
            res = tree_search.search_dfs(solution,value_model is not None,
                                         upper_bound, ts_param_p, ts_param_d, ts_param_n, verbose, val_timeout, bstrategy)
        elif sstrategy == 'lds':
            res = tree_search.search_lds(solution,value_model is not None, upper_bound,
                                             ts_param_p, ts_param_d, ts_param_n, verbose,
                                             val_timeout, bstrategy, use_bins, nbins,
                                             zero_depth)
        elif sstrategy == 'wbs':
            res = tree_search.search_wbs(solution,value_model is not None, upper_bound,
                                             ts_param_p, ts_param_d, ts_param_d2, ts_param_n, verbose,
                                             val_timeout, bstrategy, ts_param_e)
        else:
            print("Invalid search strategy: {0}".format(sstrategy)) # should never happen...
            sys.exit(1)
        move_list, nodes_count, cpu_time = res
#         rt = time.time() - start_time_test
        refMoveCount = len(rr)
        solutionMoveCount = len(move_list)
        moves_sum += solutionMoveCount
        rt_sum += cpu_time
        ref_moves_sum += refMoveCount
        ref_rt_sum += ref_rt

        solution = solution_cpmp(cells, stacks, tiers, dummy_tiers, branch_network, value_model)
        for m in move_list:
            solution.apply(m)
        if solution.is_complete() and verbose >= 2:
            print("Solution is valid")
        elif solutionMoveCount != 0:
           logging.info("CRITICAL: An invalid solution was generated.")

        if solutionMoveCount == 0:
            unsolved += 1
            if verbose >= 1:
                print("[sol] {3}; Reference: {0}; DNN: unsolved; Gap: -; CPU: {1:.2f}; Nodes: {2}".format(refMoveCount, cpu_time, nodes_count, ff))
        else:
            solved += 1
            gap = (solutionMoveCount - refMoveCount) / refMoveCount
            if verbose >= 1:
                print("[sol] {5}; Reference: {0}; DNN: {1}; Gap: {2:.2f}%; CPU: {3:.2f}; Nodes: {4}".format(refMoveCount, solutionMoveCount, gap * 100, cpu_time, nodes_count, ff))
#             #Write solution to disk TODO change this to be CLI output
            if not tuning:
                with open(os.path.join(output_path,"solutions",  ff.split('.')[0] + ".txt"), "w") as f:
                    for i in range(len(move_list)):
                        f.write(str(move_list[i])+"\n")

        results.append([ddir.split('/')[-1],ff.split('.')[0], solutionMoveCount, cpu_time, "?",nodes_count])

    if verbose >= 1:
        print("{0}% of {1} instances were solved!".format(round(solved / (solved + unsolved) * 100, 2),solved + unsolved))
        print("Average number of moves of dnn: {0}".format(round(moves_sum / solved,2)))
        print("Average number of moves of reference solution: {0}".format(round(ref_moves_sum / (solved+unsolved), 2)))
    elif tuning:
        print(solved)
        print(moves_sum)
    logging.info("Number of instances processed: {0}".format(solved + unsolved))

    return (solved, unsolved,moves_sum,ref_moves_sum,rt_sum, ref_rt_sum, results)
