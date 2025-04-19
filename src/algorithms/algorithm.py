from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Abstract class used for the different simulation runs.
    Inherit from this class, implement the run method and
    run simulations using the Runner class.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def run(self, problem, optimum, max_evals, eps=0):
        """
        Run the algorithm.
        Args:
            problem (callable): A function that evaluates a candidate solution.
            optimum (int or float): The desired target fitness value.
            max_evals (int): Maximum number of function evaluations.
            eps (float, optional): Convergence tolerance (if applicable).
        Returns:
            tuple: (best fitness found, number of evaluations used).
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Returns:
            str: A description of the algorithm.
        """
        pass
