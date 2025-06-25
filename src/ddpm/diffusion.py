import torch


class SimpleDiffusion:  # noqa: D101
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        """Initialize the noise schedule."""
        ##### Exercise 1a #####
        raise NotImplementedError("Implement the variance schedule in Simple Diffusion.")
        self.beta = self.get_betas()
        self.alpha = None
        self.sqrt_beta = None
        self.alpha_cumulative = None
        self.sqrt_alpha_cumulative = None
        self.one_by_sqrt_alpha = None
        self.sqrt_one_minus_alpha_cumulative = None
        ##### Exercise 1a #####

    def get_betas(self, beta_start=1e-4, beta_end=0.02):
        """Linear schedule, proposed in original ddpm paper.

        Parameters
        ----------
            beta_start : float, optional
                The starting value of beta, by default 1e-4.
            beta_end : float, optional
                The ending value of beta, by default 0.02.

        Returns
        -------
        torch.Tensor
            A tensor containing the beta values for each diffusion timestep.
            The shape of the tensor is (num_diffusion_timesteps,).
        """
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = beta_start * scale
        beta_end = beta_end * scale
        ##### Exercise 1a #####
        raise NotImplementedError("Implement the beta schedule in the `get_betas` function.")
        ##### Exercise 1a #####


def get(element: torch.Tensor, t: torch.Tensor):
    """Get value at index position "t" in "element" and reshape it to have the same dimension as a batch of images."""
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)


def forward_diffusion(diffusion: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    """Performs the forward diffusion process for a given batch of images and timesteps.

    This function adds noise to the input images `x0` according to the diffusion process
    defined by the `SimpleDiffusion` object and the specified timesteps. It returns the
    noised images and the noise used for the process.

    Parameters
    ----------
        diffusion : SimpleDiffusion)
            The diffusion process parameters, containing precomputed
            schedules for sqrt(alpha_cumulative) and sqrt(1 - alpha_cumulative).
        x0 : torch.Tensor
            The original images to be noised, of shape (batch_size, ...).
        timesteps : torch.Tensor
            The timesteps at which to apply the forward diffusion,
            of shape (batch_size,) or broadcastable to x0.

    Returns
    -------
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The noised images at the given timesteps.
            - The noise tensor used to generate the noised images.
    """
    ##### Exercise 1b #####
    # Hint: use get to index and reshape the noise schedule tensors
    # Sample noise

    # Scale image and noise according to the diffusion process

    # Add noise to the scaled image to get the noised sample

    # Model will learn to predict the noise from the noised sample.
    # Return (noised sample, ground truth noise)

    raise NotImplementedError("Implement the forward diffusion process in the `forward_diffusion` function.")
    ##### Exercise 1b #####
