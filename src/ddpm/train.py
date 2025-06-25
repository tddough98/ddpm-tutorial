from torch import amp
from torchmetrics import MeanMetric
from tqdm.auto import tqdm

from .config import BaseConfig, TrainingConfig


# Algorithm 1: Training
def train_one_epoch(  # noqa: D103
    model,
    diffusion,
    loader,
    optimizer,
    scaler,
    epoch=800,
    base_config=BaseConfig(),
    training_config=TrainingConfig(),
):
    loss_record = MeanMetric()
    model.train()
    ##### Exercise 2 #####
    #  Set up the loss function.
    ##### Exercise 2 #####

    with tqdm(total=len(loader), dynamic_ncols=True, leave=False) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
        for x0s, _ in loader:
            tq.update(1)
            max_timesteps = training_config.TIMESTEPS
            ##### Exercise 2 #####
            # Sample random timesteps for each image in the batch.

            # Sample noised images from the forward diffusion process.

            with amp.autocast(base_config.DEVICE.type):
                raise NotImplementedError("Implement the loss in the `train_one_epoch` function.")
                # Predict the noise in the noised images using the model
                # Compute the loss between the predicted noise and the ground truth noise.
            ##### Exercise 2 #####

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_value = loss.detach().item()
            loss_record.update(loss_value)
            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
        mean_loss = loss_record.compute().item()
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    return mean_loss
