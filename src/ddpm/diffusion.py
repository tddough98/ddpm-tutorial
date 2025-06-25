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
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)
        ##### Exercise 1a #####

    def get_betas(self, beta_start=1e-4, beta_end=0.02):
        """Linear schedule, proposed in original ddpm paper"""
        ##### Exercise 1a #####
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * beta_start
        beta_end = scale * beta_end
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
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
    eps = torch.randn_like(x0)  # Noise
    mean = get(diffusion.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
    std_dev = get(diffusion.sqrt_one_minus_alpha_cumulative, t=timesteps)  # Noise scaled
    sample = mean + std_dev * eps  # scaled inputs * scaled noise

    return sample, eps  # return ... , gt noise --> model predicts this)
    ##### Exercise 1b #####
