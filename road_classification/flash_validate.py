import torch
import pytorch_lightning as pl
import flash
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from flash.image import ImageClassificationData, ImageClassifier
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.io.input_transform import InputTransform
from dataclasses import dataclass
from torchvision import transforms as T
from typing import Tuple, Union

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

class BinaryImageClassifier(ImageClassifier):
    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        x = super().to_metrics_format(x)
        return torch.argmax(x, dim=1)

if __name__ == '__main__':
    # 1. Prep the data
    #datamodule = ImageClassificationData.from_csv(
    #    "image",
    #    "has_road",
    #    train_file="/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/train.csv",
    #    val_file="/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/val.csv",
    #    target_formatter=flash.core.data.utilities.classification.SingleLabelTargetFormatter(labels=["no", "yes"]),
    #    transform=ImageClassificationInputTransform,
    #    transform_kwargs={"image_size": (500, 500)},
    #    batch_size=8,
    #    num_workers=4,
    #)
    
    # get the heads available for ImageClassification
    heads = ImageClassifier.available_heads()
    
    # print the heads
    print(heads)

    # 2. Build the model
    metrics = [BinaryAccuracy(), BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
    model = BinaryImageClassifier.load_from_checkpoint(
        "/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/lightning_logs/version_5654180/checkpoints/epoch=79-step=33760.ckpt",
        metrics=metrics
    )
    
    print(model)

    ## 3. Create the trainer, finetune and validate the model
    #trainer = flash.Trainer(
    #    accelerator='gpu', 
    #    strategy="ddp_find_unused_parameters_false", 
    #    devices=-1, 
    #    callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=80)]
    #)
    #trainer.validate(model, datamodule=datamodule)
