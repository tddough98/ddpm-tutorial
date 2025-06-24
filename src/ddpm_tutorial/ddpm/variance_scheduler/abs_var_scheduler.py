from abc import ABC, abstractmethod


class Scheduler(ABC):  # noqa: D101
    @abstractmethod
    def get_alpha_hat(self):  # noqa: D102
        pass

    @abstractmethod
    def get_alphas(self):  # noqa: D102
        pass

    @abstractmethod
    def get_betas(self):  # noqa: D102
        pass

    @abstractmethod
    def get_betas_hat(self):  # noqa: D102
        pass
