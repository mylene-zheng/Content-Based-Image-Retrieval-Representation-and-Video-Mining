# Content-Based Image Retrieval & Shot-Cut Detection

Image and video mining coursework, in one Jupyter notebook with two parts:

1. **Content-based image retrieval (CBIR):** given a query image, rank 150 frames sampled from Hitchcock's *Vertigo* dream scene by visual similarity. The notebook compares colour histograms, ResNet-50 and DINOv2 features.
2. **Shot-cut detection:** find the hard cuts in a video. The detector is tuned on *The Matrix* (development set), tested unchanged on *Cosmos Laundromat* (held-out set), and then checked on three more clips that I annotated by hand.

All results are computed and checked inside the notebook rather than read off the figures. Each choice of method is explained next to the code that implements it.

---

## Key results

### Part 1: Image retrieval (4 queries, pass/fail criteria checked automatically)

| Representation | Similarity | Queries passed |
|---|---|:---:|
| RGB colour histogram (512-D) | cosine / intersection / Bhattacharyya / χ² | 2 / 4 |
| ResNet-50, official preprocessing (224 centre crop) | cosine | 3 / 4 |
| **ResNet-50, native-scale preprocessing** | cosine | **4 / 4** |
| DINOv2 ViT-S/14 (224 centre crop) | cosine | 2 / 4 |

The four queries get harder in order: a whole gallery frame, a crop of a gallery frame, a crop of a frame held out of the gallery, and an external painting (*Portrait of a Lady*, Zaganelli).

**Takeaways**
- The representation decides *what kind* of similarity is retrieved. Colour histograms return frames from the same shot, and deep features return frames with the same content. The similarity measure mostly reshuffles near-ties.
- The biggest gain came from **getting the preprocessing scale right**. The standard 224 px centre crop enlarges a crop query about 1.6× relative to the gallery frames and throws away 45 % of each 16:9 frame. Feeding every image to ResNet-50 at the video's pixel scale fixes both problems.
- Only the deep models bridge the gap between the painting and the film frames. They retrieve the scene of a woman wearing a pendant, while the histogram retrieves dark-red frames of a man.

### Part 2: Shot-cut detection (one-to-one matching, ±1 frame tolerance)

| Detector | Matrix F1 (dev) | Cosmos (held-out) TP / FP / FN | Cosmos F1 |
|---|:---:|:---:|:---:|
| Baseline: 1 − cosine on RGB histogram, threshold + local max | 0.933 | 26 / 69 / 5 | 0.413 |
| **Improved: χ² on HSV histogram + temporal dominance test** | **1.000** | **29 / 2 / 2** | **0.935** |

- **Temporal dominance:** a cut must be an *isolated* spike, scoring at least α times the maximum score in its ±6-frame neighbourhood. This removes bursts of false alarms caused by flashes, flicker and colour pulses.
- **Parameters fitted on Matrix only:** T and α are each set to the geometric midpoint of the gap between cuts and non-cuts on Matrix. A sensitivity sweep shows Cosmos F1 stays between 0.90 and 0.97 for T from 0.08 to 0.22 and α from 1.3 to 2.5.
- The notebook also includes an ablation of each component, storyboards of every shot (with a representative frame chosen as the frame closest to the shot's mean histogram), and a comparison with z-score thresholds and "n frames before vs. n frames after" scores.

**Robustness check (hand-annotated clips).** The detector does *not* transfer well to footage that differs from the development clip. Pooled F1 over the three extra clips is 0.495 (baseline: 0.336), and the notebook explains each failure:
- On black-and-white film, the HSV histogram reduces to 4 brightness bins.
- Frame-rate conversion leaves blended frames that split a cut across two boundaries.
- In *Vertigo*, cuts are deliberately placed next to colour-light pulses.

---

## Repository contents

| Path | Description |
|---|---|
| [`CBIR_ShotCut_Detection.ipynb`](CBIR_ShotCut_Detection.ipynb) | Main notebook: code, figures, results and discussion |
| [`Report_CBIR_ShotcutDetection .pdf`](Report_CBIR_ShotcutDetection%20.pdf) | Written report |
| [`videos_clips/`](videos_clips/) | The five input clips (Cosmos Laundromat, Man with a Movie Camera, Vertigo, Entr'acte, The Matrix) |
| [`Portrait_of_a_Lady.jpg`](Portrait_of_a_Lady.jpg) | External query image for Part 1 |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |

---

## How to run

Tested with **Python 3.13.5, CPU only** (no GPU needed). Run the notebook from the repository folder, because all paths are relative.

```bash
git clone https://github.com/mylene-zheng/Content-Based-Image-Retrieval-Representation-and-Video-Mining.git
cd Content-Based-Image-Retrieval-Representation-and-Video-Mining

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab CBIR_ShotCut_Detection.ipynb
```

On the first run, the ResNet-50 `IMAGENET1K_V2` weights (torchvision) and DINOv2 ViT-S/14 (`torch.hub`) are downloaded automatically, so an internet connection is needed. If DINOv2 can't be downloaded, the notebook skips it and runs everything else.

**Main libraries:** PyAV (video decoding), NumPy, Pillow, Matplotlib, PyTorch / torchvision, ipywidgets (optional interactive shot explorer).

---

## Notes

- **Evaluation protocol.** All detector parameters are frozen on *Matrix* before *Cosmos* is decoded, and the cell order enforces this. The notebook states one limitation openly: the baseline's errors on Cosmos were inspected while the improved pipeline was being designed.
- **Data.** The video clips and the painting were provided as course material and belong to their respective rights holders. They are included only so the notebook can be reproduced.
