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

    def initialize(self):  # noqa: D102
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """Linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )


def get(element: torch.Tensor, t: torch.Tensor):
    """Get value at index position "t" in "element" and reshape it to have the same dimension as a batch of images."""
    ele = element.gather(-1, t)
    return ele.reshape(-1, 1, 1, 1)


def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):  # noqa: D103
    eps = torch.randn_like(x0)  # Noise
    mean = get(sd.sqrt_alpha_cumulative, t=timesteps) * x0  # Image scaled
    std_dev = get(sd.sqrt_one_minus_alpha_cumulative, t=timesteps)  # Noise scaled
    sample = mean + std_dev * eps  # scaled inputs * scaled noise

    return sample, eps  # return ... , gt noise --> model predicts this)
