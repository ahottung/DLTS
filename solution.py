from abc import ABC, abstractmethod

class solution(ABC):

    @abstractmethod
    def apply(self, move):
        pass

    @abstractmethod
    def undo_last_move(self):
        pass

    @abstractmethod
    def is_complete(self):
        pass

    @abstractmethod
    def get_cost(self):
        pass

    @abstractmethod
    def get_illegal_moves(self):
        pass

    @abstractmethod
    def get_move_list(self):
        pass

    @abstractmethod
    def get_branch_network_prediction(self):
        pass

    @abstractmethod
    def get_lb_network_prediction(self):
        pass
