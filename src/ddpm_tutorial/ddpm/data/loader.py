import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm.auto import tqdm


def _get_loader(config):
    dataset = load_dataset(config.dataset, split="train")
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images_key = "image" if "image" in dataset.features else "img"

    mode = "L" if config.image_channels == 1 else "RGB"

    def transform(examples):
        images = [preprocess(image.convert(mode)) for image in examples[images_key]]
        return {"images": images}

    # Precompute all transformations
    transformed_dataset = dataset.map(
        transform,
        remove_columns=[col for col in dataset.column_names if col != "images"],
    )

    loader = DataLoader(transformed_dataset, batch_size=config.train_batch_size, shuffle=True)
    return loader


def get_loader(config):
    """Create a DataLoader with fully precomputed tensors for maximum training performance.

    All preprocessing is done once upfront, then training just loads tensors.
    """
    # Load dataset
    print(f"Loading dataset: {config.dataset}")
    dataset = load_dataset(config.dataset, split="train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")

    # Find the image column name
    image_key = None
    for key in ["image", "img", "pixel_values"]:
        if key in dataset.features:
            image_key = key
            break
    if image_key is None:
        raise ValueError(f"Could not find image column. Available: {list(dataset.features.keys())}")
    print(f"Using image key: {image_key}")

    # Define preprocessing pipeline (deterministic only, no random transforms)
    preprocess_transforms = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Preprocess images and store as tensors
    print("Preprocessing all images (this may take a moment)...")
    all_images = []
    batch_size_preprocess = 1000
    total_items = len(dataset)
    for i in tqdm(range(0, total_items, batch_size_preprocess), desc="Preprocessing"):
        end_idx = min(i + batch_size_preprocess, total_items)
        batch_data = dataset[i:end_idx]
        batch_images = []
        for image in batch_data[image_key]:
            processed_image = preprocess_transforms(image)
            batch_images.append(processed_image)
        all_images.extend(batch_images)

    # Convert to tensors
    print("Converting to tensors...")
    images_tensor = torch.stack(all_images)
    print(f"Images tensor shape: {images_tensor.shape}")
    # if preload:
    #     print("Preloading tensors into memory...")
    #     images_tensor = images_tensor.to(config.device)
    tensor_dataset = TensorDataset(images_tensor)
    # Create DataLoader with precomputed tensors
    print("Creating DataLoader...")
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=True,  # Keep tensors in memory for faster access
        num_workers=4,
        persistent_workers=True,  # Keep workers alive for faster subsequent epochs
        prefetch_factor=2,  # Prefetch 2 batches in advance
    )

    print(f"DataLoader created successfully! Memory usage: ~{images_tensor.numel() * 4 / 1024**3:.2f} GB")
    return dataloader
