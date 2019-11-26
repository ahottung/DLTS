from solution import solution
import numpy as np
import copy

def get_stacks_height(cells, stacks, tiers):
    height = []
    for s in range(stacks):
        for t in range(tiers):
            if cells[s][t] == 0:
                height.append(t)
                break
            if t == tiers-1:
                height.append(tiers)
    return height

def is_sorted(cells, stacks, tiers):
    for s in range(stacks):
        for t in range(tiers-1):
            if(cells[s][t] < cells[s][t+1]):
                return False
    return True

class solution_cpmp(solution):
    cells = []
    height = []
    stacks = -1
    tiers = -1
    dummy_tiers = 0
    move_list = []
    branch_network = None
    lb_network = None
    complete = None # Only use with LDS/BFS!!!

    def __init__(self, start_state, stacks, tiers, dummy_tiers, branch_network, lb_network=None):
        self.cells = np.reshape(start_state, (stacks, -1))
        self.stacks = stacks
        self.tiers = tiers
        self.dummy_tiers = dummy_tiers
        self.height = get_stacks_height(self.cells,self.stacks,self.tiers)
        self.branch_network = branch_network
        self.lb_network = lb_network
        self.move_list = []
        self.complete = None

    def apply(self,move):
        mFrom = int(move / (self.stacks - 1))
        mTo = move % (self.stacks - 1)
        if mTo >= mFrom:
            mTo += 1

        self.cells[mTo][self.height[mTo]] = self.cells[mFrom][self.height[mFrom] - 1]
        self.cells[mFrom][self.height[mFrom] - 1] = 0
        self.height[mTo] += 1
        self.height[mFrom] -= 1
        self.move_list.append(move)

    def undo_last_move(self):
        move = self.move_list[len(self.move_list)-1]
        mFrom = int(move / (self.stacks - 1))
        mTo = move % (self.stacks - 1)
        if mTo >= mFrom:
            mTo += 1

        self.cells[mFrom][self.height[mFrom]] = self.cells[mTo][self.height[mTo] - 1]
        self.cells[mTo][self.height[mTo] - 1] = 0
        self.height[mFrom] += 1
        self.height[mTo] -= 1
        self.move_list.pop()

    def is_complete(self):
        return is_sorted(self.cells,self.stacks,self.tiers)

    def is_complete_memo(self):
        if not self.complete:
            self.complete = is_sorted(self.cells,self.stacks,self.tiers)
        return self.complete

    def get_cost(self):
        return len(self.move_list)

    # TODO could include symmetries, etc. here
    def get_illegal_moves(self):
        illegal_moves = []
        last_mTo = -1
        if len(self.move_list) > 0:
            last_move = self.move_list[len(self.move_list)-1]
            last_mTo = last_move % (self.stacks - 1)
            if last_mTo >= int(last_move / (self.stacks - 1)):
                last_mTo += 1

        for i in range(self.stacks*(self.stacks-1)):
            mFrom = int(i / (self.stacks - 1))
            mTo = i % (self.stacks - 1)
            if mTo >= mFrom:
                mTo += 1

            if self.height[mTo] == self.tiers or self.height[mFrom] <= self.dummy_tiers or last_mTo == mFrom:
                illegal_moves.append(i)

        return illegal_moves

    def get_move_list(self):
        return self.move_list

    f1 = None
    def get_branch_network_prediction(self):
#         #x = np.hsplit(np.array([np.reshape(self.cells, -1)]), self.stacks)
#         #z = [np.reshape(self.cells, -1)]
#         #y = np.array([np.reshape(self.cells, -1)])
#         #zz = np.reshape(self.cells,(1,35))
#         #return self.branch_network.predict(np.hsplit(np.array([np.reshape(self.cells, -1)]), self.stacks))[0]
#         #return self.branch_network.predict(np.hsplit(np.reshape(self.cells, (1,35)), self.stacks))[0]

        # 45% faster
        if self.f1 is None:
            self.branch_network.predict(np.hsplit(np.array([np.reshape(self.cells, -1)]).astype('float32'), self.stacks))[0]
#             #z = np.memmap(np.hsplit(np.array([np.reshape(self.cells, -1)]), self.stacks), dtype='float32')
#             #self.branch_network.predict(z)[0]
            self.f1 = self.branch_network.predict_function
            self.f1.function.trust_input = True
        batch_outs = self.f1(np.hsplit(np.array([np.reshape(self.cells, -1)]).astype('float32'), self.stacks))
        return batch_outs[0][0]

    f2 = None
    def get_lb_network_prediction(self):
#         #return self.lb_network.predict(np.hsplit(np.array([np.reshape(self.cells, -1)]), self.stacks))[0][0]
#         #return self.lb_network.predict(np.hsplit(np.reshape(self.cells,(1,35)), self.stacks))[0][0]

        if self.f2 is None:
            self.lb_network.predict(np.hsplit(np.array([np.reshape(self.cells, -1)]).astype('float32'), self.stacks))[0][0]
            self.f2 = self.lb_network.predict_function
            self.f2.function.trust_input = True
        batch_outs = self.f2(np.hsplit(np.array([np.reshape(self.cells, -1)]).astype('float32'), self.stacks))
        return batch_outs[0][0][0]

    def __deepcopy__(self, memo):
        cells = copy.deepcopy(self.cells)
        stacks = self.stacks
        tiers = self.tiers
        dummy_tiers = self.dummy_tiers
        move_list = copy.deepcopy(self.move_list)
        branch_network = self.branch_network
        lb_network = self.lb_network
        ret = solution_cpmp(cells, stacks, tiers, dummy_tiers, branch_network, lb_network)
        ret.move_list = move_list
        return ret
