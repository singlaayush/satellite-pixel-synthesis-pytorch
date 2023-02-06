import torch
import pytorch_lightning as pl
import flash
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from flash.image import ImageClassificationData, ImageClassifier
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.io.input_transform import InputTransform
from dataclasses import dataclass
from torchvision import transforms as T
from typing import Callable, Tuple, Union
from torchdependencies import RoadDataset, make_weights_for_balanced_classes

# Function for setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# unused for now
@dataclass
class ImageClassificationInputTransform(InputTransform):
    image_size: Tuple[int, int] = (500, 500)
    mean: Union[float, Tuple[float, float, float]] = (0.4300, 0.3860, 0.3388)
    std:  Union[float, Tuple[float, float, float]] = (0.1870, 0.1533, 0.1267)

    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose([T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std)]),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.Resize(self.image_size),
                            T.Normalize(self.mean, self.std),
                            T.RandomHorizontalFlip(),
                            T.ColorJitter(),
                            T.RandomAutocontrast(),
                            T.RandomPerspective(),
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

class BinaryImageClassifier(ImageClassifier):
    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        x = super().to_metrics_format(x)
        return torch.argmax(x, dim=1)

if __name__ == '__main__':
    # Create training and validation datasets
    image_datasets = {x: RoadDataset(mode=x, transform=None) for x in ['train']}
    weights, weight_per_class = make_weights_for_balanced_classes(image_datasets['train'], 2)
    weights = torch.DoubleTensor(weights)
    print(f"Class-wise weights calculated for sampler: {weight_per_class}")
    
    # 1. Prep the data
    datamodule = ImageClassificationData.from_csv(
        "image",
        "has_road",
        train_file="/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/train.csv",
        val_file="/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/val.csv",
        target_formatter=flash.core.data.utilities.classification.SingleLabelTargetFormatter(labels=["no", "yes"]),
        transform=ImageClassificationInputTransform,
        transform_kwargs={"image_size": (500, 500)},
        batch_size=8,
        sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True),
        num_workers=4,
    )
    #    #transform_kwargs={
    #    #    "image_size": (500, 500),
    #    #    "mean": (0.4300, 0.3860, 0.3388), 
    #    #    "std": (0.1870, 0.1533, 0.1267)
    #    #},

    # 2. Build the model
    metrics = [BinaryAccuracy(), BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
    model = ImageClassifier(
        backbone="densenet121", 
        labels=datamodule.labels, 
        optimizer="Adam",
        learning_rate=1e-4, 
        metrics=metrics,
        pretrained=False
    )
    model.configure_optimizers()
    
    print(f"Learning rate set to {model.learning_rate}.")
        
    # 3. Create the trainer, finetune and validate the model
    trainer = flash.Trainer(
        max_epochs=201, 
        accelerator='gpu', 
        strategy="ddp_find_unused_parameters_false", 
        devices=-1, 
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=80)]
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
