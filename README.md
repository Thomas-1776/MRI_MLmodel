# Brain MRI Tumor Detection (U-Net, patient-level evaluation)

Slice-level tumor detection and segmentation on the LGG brain MRI dataset, using a U-Net with a pretrained ResNet18 encoder (PyTorch).

> Research / educational project. Not a medical device and not for clinical use.

## Highlights

- **U-Net reaches 93.3% accuracy and 95.9% tumor recall on 17 held-out patients**, against 82.9% accuracy and 70.8% recall for a frozen-CNN-features + SVM baseline evaluated the same way (patient-grouped).
- **Found and fixed a data leakage problem.** The baseline's first random slice-level split put slices of the same patient in both train and test (all 110 patients overlapped), which inflated accuracy to about 93%. Re-evaluating with patient-grouped splits dropped it to 82.9%.
- Small tumors are the weak spot (86-89% recall); medium and large tumors are caught almost every time.

## Dataset

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (from TCGA-LGG): 3,929 brain MRI slices from 110 patients, each with a tumor mask. A slice is labeled "tumor" if its mask contains any positive pixel.

Neighbouring slices of one patient are nearly identical, so all splits are done **by patient**, never by slice.

## Approach

**1. Baseline (`svm_baseline.ipynb`):** frozen MobileNetV2 + DenseNet121 features (2,304-D) fed to an RBF SVM. Masks are used only to create the yes/no label.

**2. U-Net (`unet_mri.py`):**
- Encoder: ImageNet-pretrained ResNet18; decoder: standard U-Net upsampling path with skip connections.
- Trained on the masks with BCE + Dice loss, horizontal-flip and brightness augmentation, AdamW with cosine learning-rate decay, 25 epochs, best checkpoint chosen by validation loss.
- Patient-grouped split: about 70% train / 15% validation / 15% test (17 test patients, 533 test slices: 196 tumor, 337 healthy).
- Slice decision rule: a slice is called "tumor" if the predicted mask has at least N pixels. N is chosen on the validation set only, then frozen for the test set.

## Results

Test set: 17 unseen patients, 533 slices.

| Model | Accuracy | Tumor recall | Specificity | Precision |
|---|---|---|---|---|
| SVM baseline (patient-grouped 5-fold CV, mean ± std) | 82.85% ± 0.52 | 70.79% ± 5.53 | - | - |
| U-Net, 128×128 | **93.25%** | **95.92%** | 91.69% | 87.04% |
| U-Net, 224×224 | 92.50% | 95.41% | 90.80% | 85.78% |

| U-Net run | Min. pixels | Dice (tumor slices) | TP | TN | FP | FN | Recall: small / medium / large tumors |
|---|---|---|---|---|---|---|---|
| 128×128 | 5 | 0.755 | 188 | 309 | 28 | 8 | 89.4% / 98.5% / 100% |
| 224×224 | 20 | 0.796 | 187 | 306 | 31 | 9 | 86.4% / 100% / 100% |

Notes:
- Detection is essentially the same at both resolutions (the differences are a few slices). 224×224 gives a better outline (Dice 0.755 to 0.796) but did not improve small-tumor detection.
- The U-Net numbers come from a single patient-level split with 17 test patients, so treat them as approximate. The SVM baseline was cross-validated over all 110 patients, so the two are not perfectly like-for-like, though the gap is large.
- For reference, the SVM scored about 93% with the leaky random split. That number is not valid and is kept only to show the effect of leakage.

## Repository contents

| File | Description |
|---|---|
| `unet_mri.py` | Data loading, patient-grouped split, U-Net training and evaluation |
| `unet_traced.pt` | Trained model exported with TorchScript (architecture + weights, stored with Git LFS) |
| `results_128.json`, `results_224.json` | Test metrics for each run |
| `svm_baseline.ipynb` | Frozen-features + SVM baseline and the leakage check |

## Using the trained model

The model file is stored with Git LFS, so clone with Git LFS installed (`git lfs install`, then `git clone ...`). GitHub's "Download ZIP" does not include LFS files.

```python
import torch, numpy as np
from PIL import Image

model = torch.jit.load("unet_traced.pt", map_location="cpu").eval()

IMG, MIN_PIXELS = 224, 20          # settings of the 224x224 run
img = Image.open("slice.tif").convert("RGB").resize((IMG, IMG), Image.BILINEAR)
x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float()   # keep 0-255

with torch.no_grad():
    mask = torch.sigmoid(model(x))[0, 0].numpy() > 0.5

print("tumor" if mask.sum() >= MIN_PIXELS else "no tumor")
```

Input must be a 3-channel brain slice like those in the dataset, resized to the size the model was traced with. The model normalizes the input internally.

## Training your own

1. Download the dataset and place the `kaggle_3m` folder where `unet_mri.py` can find it (it auto-detects `/kaggle/input` and `/content/data`; otherwise set `LOCAL_PATH`).
2. Run `python unet_mri.py`. A GPU is strongly recommended (about 8 seconds per epoch on a Colab T4).
3. Settings at the top of the script: `IMG`, `EPOCHS`, `BS`, `LR`.

## Limitations

- One split with 17 test patients; no confidence intervals for the U-Net.
- Tested only on this dataset (one source, FLAIR-style slices). No external validation, so performance on other scanners or sequences is unknown.
- Small tumors are missed more often (86-89% recall).
- Slice-level labels only; no 3D context across neighbouring slices.

## Possible next steps

- 5-fold patient-grouped cross-validation for the U-Net (mean ± std over all 110 patients).
- Threshold sweep for a recall/false-alarm trade-off curve.
- Export to ONNX and INT8 quantization to measure size, latency and accuracy cost for deployment on smaller hardware.
