import numpy as np
import time
import copy
import math
from heapq import *
import sys

class DFSTreeNode(object):
    def __init__(self, max_prob, branches):
        self.max_prob = max_prob
        self.branches = branches

class LDSTreeNode(object):
    def __init__(self, discrepency, depth, pre_solution, branch_num):
        self.discrepency = discrepency
        self.depth = depth
        self.pre_solution = pre_solution
        self.branch_num = branch_num
#         self.sort_value = (depth / start_ub) * discrepency

    def __lt__(self, other):
        # don't do this
#         if self.solution.is_complete_memo():
#             return True
#         elif other.solution.is_complete_memo():
#             return False
#         if self.sort_value == other.sort_value:
#             return self.depth > other.depth
        if self.discrepency == other.discrepency:
            return self.depth > other.depth # could parameterize, but I think > should be better
        return self.discrepency < other.discrepency

class BFSTreeNode(object):
    def __init__(self, depth, lb, solution):
        self.depth = depth
        self.lb = lb
        self.solution = solution

    def __lt__(self, other):
        if self.lb == other.lb:
            return self.depth < other.depth ##maybe we should change this to >
        return self.lb < other.lb

def branch_strategy_constant(mp, ub, dd, pp):
    """
    (Some parameter may not be used in this branching strategy)
    mp -- maximum probability coming out of the DNN
    ub -- number of moves in current best known solution
    dd -- current search depth
    pp -- search width parameter
    Returns:
    """
    return mp * (1 - pp)

def branch_strategy_linear(mp, ub, dd, pp):
    return mp * (1 - pp * (dd / ub))

def branch_strategy_quadratic(mp, ub, dd, pp):
    return mp * (1 - pp * ((ub - dd)**2 / ub**2))

def branch_strategy_log(mp, ub, dd, pp):
    return mp * (1 - pp * (-np.log(dd / ub)))

def search_dfs(solution, use_lb_network, start_ub, pp, dd=0.95, nn=3, verbose=1, timeout=60.0,
               bstrategy='log'):
    start_time = time.process_time()
    stack = []
    ub = start_ub
    incumbent = []
    root = True
    node_count = 0

    branch_func = branch_strategy_constant
    if bstrategy == 'linear':
        branch_func = branch_strategy_linear
    elif bstrategy == 'quadratic':
        branch_func = branch_strategy_quadratic
    elif bstrategy == 'log':
        branch_func = branch_strategy_log

    while (stack or root) and (time.process_time() - start_time < timeout):
        root = False
        node_count += 1
        ### Check for solution
        # If a solution is complete check for a new incumbent
        if solution.is_complete():
            cost = solution.get_cost()
            if ub > cost:
                ub = cost
#                 print("[search] New incumbent: {0}".format(ub))
                incumbent = solution.get_move_list().copy()
                if verbose >= 2:
                    print("[dfs] {1:.2f}s New incumbent: {0}".format(ub, (time.process_time() - start_time)))
#                 print(str(cost))

        ### Check lower bound
        depth = solution.get_cost()
        # Use the lower bound network all n levels of the tree to predict the number of remaining moves
        # If the number of predicted moves * d + made moves is higher than the lower bound the node is cut off
        # The paramter d is used because of the imprecision of the prediction
        if depth + 1 >= ub or \
                (use_lb_network and depth % nn == 0 and
                 depth + (solution.get_lb_network_prediction() * dd) + 1 > ub):
            while stack:
                solution.undo_last_move()
                tn = stack[-1]
                branch_values = tn.branches
                max_prob = tn.max_prob
                move = np.argmax(branch_values)
                if branch_values[move] > -1 and \
                        branch_values[move] >= branch_func(max_prob, ub, solution.get_cost(), pp):
                    move = np.argmax(branch_values)
                    branch_values[move] = -1
                    solution.apply(move)
                    break # reason: we found a move to apply and now want to go back down the tree
                stack.pop()

        else:
            ### Branch
            # Make a prediction for each move (between 0-1, all predictions add up to 1)
            branch_values = solution.get_branch_network_prediction()
            # Remove illegal moves
            illegal_moves = solution.get_illegal_moves()
            branch_values[illegal_moves] = -1
            move = np.argmax(branch_values)
            tn = DFSTreeNode(branch_values[move], branch_values)
            stack.append(tn)
            branch_values[move] = -1
            solution.apply(move)

#     #print(str(node_counter))
    return (incumbent, node_count, time.process_time() - start_time)

def push_children_lds(heap, node, solution, branch_func, ub, pp, use_bins, nbins, zero_depth):
    """
    Pre: node.depth + 1 < ub
    Post: all feasible child nodes not exceeding the heuristic LB are added to the heap
    """
    branch_values = solution.get_branch_network_prediction()
    illegal_moves = solution.get_illegal_moves()
    branch_values[illegal_moves] = -1
    max_prob = max(branch_values)
    sort_mp = [(ii, vv) for ii,vv in enumerate(branch_values) if vv > -1]
    sort_mp.sort(key=lambda xx: xx[1], reverse=True)
    new_depth = node.depth + 1
    if use_bins:
        bin_size = max_prob / nbins
    for disc, (move_num, mp) in enumerate(sort_mp):
        if mp >= branch_func(max_prob, ub, solution.get_cost(), pp):
            new_discrepency = disc
            if use_bins:
                new_discrepency = nbins - int(math.ceil(mp / bin_size))
            if new_depth <= zero_depth:
                new_discrepency = 0
            tn = LDSTreeNode(node.discrepency + new_discrepency, new_depth, solution, move_num)
            heappush(heap, tn)

def search_lds(start_solution, use_lb_network, start_ub, pp, dd=0.95, nn=3, verbose=1, timeout=60.0,
               bstrategy='log', use_bins=False, nbins=5, zero_depth=-1):
    start_time = time.process_time()
    incumbent = []
    node_count = 0
    ub = start_ub

    branch_func = branch_strategy_constant
    if bstrategy == 'linear':
        branch_func = branch_strategy_linear
    elif bstrategy == 'quadratic':
        branch_func = branch_strategy_quadratic
    elif bstrategy == 'log':
        branch_func = branch_strategy_log

    heap = []
    root = LDSTreeNode(0, 0, start_solution, 0)
    heappush(heap, root)

    while heap and (time.process_time() - start_time < timeout):
        cur = heappop(heap)
        solution = copy.deepcopy(cur.pre_solution)
        node_count += 1
        if cur.depth != 0:
            solution.apply(cur.branch_num)
        if solution.is_complete() and cur.depth < ub:
            ub = cur.depth
            incumbent = solution.get_move_list().copy()
            if verbose >= 2:
                print("[lds] {1:.2f}s New incumbent: {0} (discrepency: {2})".format(ub, (time.process_time() - start_time), cur.discrepency))
        elif cur.depth + 1 < ub:
            if not use_lb_network or cur.depth % nn != 0 or cur.depth + 1 + \
                    (solution.get_lb_network_prediction() * dd) < ub:
                push_children_lds(heap, cur, solution, branch_func, ub, pp, use_bins, nbins, zero_depth)

    if verbose == 2:
        print("Heap size at end of search: {0}".format(len(heap)))
    return (incumbent, node_count, time.process_time() - start_time)

def push_children_wbs(heap, node, branch_func, ub, pp, use_lb_network, nn, dd, dd2, ee):
    """
    Pre: node.depth + 1 < ub
    Post: all feasible child nodes not exceeding the heuristic LB are added to the heap
    """
    branch_values = node.solution.get_branch_network_prediction()
    illegal_moves = node.solution.get_illegal_moves()
    branch_values[illegal_moves] = -1
    max_prob = max(branch_values)
    sort_mp = [(ii, vv) for ii,vv in enumerate(branch_values) if vv > -1]
    sort_mp.sort(key=lambda xx: xx[1], reverse=True)
    new_depth = node.depth + 1
    ret = 0
    incumbent = []
    for disc, (move_num, mp) in enumerate(sort_mp):
        if mp >= branch_func(max_prob, ub, node.solution.get_cost(), pp):
            scopy = copy.deepcopy(node.solution)
            scopy.apply(move_num)
            if scopy.is_complete() and new_depth < ub:
                ub = new_depth
                incumbent = scopy.get_move_list().copy()
            lb_prediction = scopy.get_lb_network_prediction()
            if not use_lb_network or new_depth % nn != 0 or new_depth + (lb_prediction * dd) < ub:
                tn = BFSTreeNode(new_depth,  (new_depth * ee) + (lb_prediction * dd2), scopy) ##maybe we should introduce a new
                # parameter here and use it instead of dd, because we might want to use a smaller value here
                heappush(heap, tn)
                ret += 1
    return (ret,ub, incumbent)

def search_wbs(start_solution, use_lb_network, start_ub, pp, dd=0.95, dd2=1.0, nn=3, verbose=1, timeout=60.0,
               bstrategy='log', ee=0.95):
    start_time = time.process_time()
    incumbent = []
    node_count = 0
    ub = start_ub

    branch_func = branch_strategy_constant
    if bstrategy == 'linear':
        branch_func = branch_strategy_linear
    elif bstrategy == 'quadratic':
        branch_func = branch_strategy_quadratic
    elif bstrategy == 'log':
        branch_func = branch_strategy_log

    heap = []
    root = BFSTreeNode(0, start_solution.get_lb_network_prediction(), start_solution)
    if start_solution.is_complete():
        incumbent = start_solution.get_move_list().copy()
    else:
        heappush(heap, root)

    while heap and (time.process_time() - start_time < timeout):
        cur = heappop(heap)
        if cur.depth + 1 < ub:
            ret, ret_ub, ret_incumbent = push_children_wbs(heap, cur, branch_func, ub, pp, use_lb_network, nn, dd, dd2, ee)
            node_count += ret
            if ret_ub < ub:
                ub = ret_ub
                incumbent = ret_incumbent
                if verbose >= 2:
                    print("[bfs] {1:.2f}s New incumbent: {0}".format(ub, (time.process_time() - start_time)))
    if verbose == 2:
        print("Heap size at end of search: {0}".format(len(heap)))
    return (incumbent, node_count, time.process_time() - start_time)
