import torch
from torch import amp
from torchmetrics import MeanMetric
from tqdm.auto import tqdm

from .config import BaseConfig, TrainingConfig
from .diffusion import forward_diffusion


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
    loss_fn = torch.nn.MSELoss()
    ##### Exercise 2 #####

    with tqdm(total=len(loader), dynamic_ncols=True, leave=False) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
        for x0s, _ in loader:
            tq.update(1)
            max_timesteps = training_config.TIMESTEPS
            ##### Exercise 2 #####
            ts = torch.randint(low=1, high=max_timesteps, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(diffusion, x0s, ts)

            with amp.autocast(base_config.DEVICE.type):
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)
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
