import torch
import torchvision
import torchvision.transforms as TF
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm

from .data import frames2vid, inverse_transform


# Algorithm 2: Sampling
@torch.inference_mode()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), num_images=5, nrow=8, device="cpu", **kwargs):  # noqa: D103
    generate_video = kwargs.get("generate_video", False)
    save_path = kwargs.get("save_path", "output.png")
    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    if generate_video:
        outs = []

    for time_step in tqdm(
        iterable=reversed(range(1, timesteps)),
        total=timesteps - 1,
        dynamic_ncols=False,
        desc="Sampling :: ",
        position=0,
        leave=False,
    ):
        ##### Exercise 3 #####
        # Reshape the time_step to match the batch size.

        # Sample noise z from a standard normal distribution.

        # Predict the noise in the current image x_t using the model.

        # Get parameters from diffusion schedule for the current timestep. (Use the `get` function to extract and reshape the values.)

        # Compute the new x using the reverse diffusion equation.
        # x = 1/sqrt(alpha_t) * (x - beta_t / sqrt(1 - alpha_cumulative_t) * predicted_noise) + sqrt(beta_t) * z
        # where alpha_t = 1 - beta_t, and alpha_cumulative_t is the cumulative product of alpha_t.

        raise NotImplementedError("Implement the reverse diffusion process in the `reverse_diffusion` function.")
        ##### Exercise 3 #####

        if generate_video:
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)

    if generate_video:  # Generate and save video of the entire reverse process.
        frames2vid(outs, save_path)
        display(
            Image.fromarray(outs[-1][:, :, ::-1])
        )  # Display the image at the final timestep of the reverse process.
        return None

    else:  # Display and save the image at the final timestep of the reverse process.
        x = inverse_transform(x).type(torch.uint8)
        grid = torchvision.utils.make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(save_path, format=save_path[-3:].upper())
        display(pil_image)
        return None
