# Secure AI Systems - Red and Blue Teaming an MNIST Classifier

```

## How to Run

```bash
pip install -r requirements.txt

python data_integrity.py
python baseline.py
python fgsm_attack.py
python poison_attack.py
python adversarial_train.py
python final_results.py

bandit -r . -x ./data,./reports -f txt -o reports/bandit.txt
pip-audit -r requirements.txt
```

The scripts use the GPU if PyTorch can find one, otherwise the CPU. On Windows, `pip install torch` gives the CPU-only version. For the GPU version (we used an RTX 3060):

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
```

MNIST downloads automatically into `data/` the first time. All scripts use `torch.manual_seed(42)`, but results on a GPU can still change slightly between runs.

Full output is in `reports/run_log.txt`.

## 4. Results

### Baseline

| Metric | Result |
|---|---:|
| Test accuracy | 98.88% |
| Test loss | 0.0323 |
| Inference time | 0.2089 ms per image (GPU) |

Confusion matrix on the clean test set (rows = actual digit, columns = predicted):

```text
[[ 977    0    0    0    0    0    2    1    0    0]
 [   0 1132    2    0    0    0    0    1    0    0]
 [   2    2 1027    0    0    0    0    1    0    0]
 [   0    0    7  998    0    1    0    1    2    1]
 [   1    0    0    0  969    0    2    2    1    7]
 [   2    0    0   11    0  870    2    1    3    3]
 [   2    2    1    0    1    1  949    0    2    0]
 [   0    2   11    0    0    0    0 1012    1    2]
 [   3    0    4    1    1    1    0    0  962    2]
 [   0    2    1    0    5    0    0    7    2  992]]
```

Most mistakes are between digits that look alike, like 5 vs 3, 7 vs 2 and 3 vs 2.

### Attack 1: FGSM (OWASP ML01 - Input Manipulation)

**Why we chose it:** Anyone who can send an image to the model can try this attack, and it doesn't need any access to the training data. FGSM is the simplest gradient attack and only takes a few lines of code, so it was easy to write ourselves instead of using a library. It also shows how a model that scores almost 99% can still be fooled by tiny changes.

FGSM adds `epsilon * sign(gradient of the loss)` to each pixel. This pushes every pixel slightly in the direction that makes the model more wrong.

| Epsilon | Adversarial accuracy | Attack success rate |
|---:|---:|---:|
| 0.05 | 95.39% | 3.53% |
| 0.10 | 85.44% | 13.59% |
| 0.20 | 40.18% | 59.36% |

Attack success rate = correctly classified images that became wrong after the attack, divided by all correctly classified images.

At epsilon 0.2 the digits are still easy for a person to read (see `reports/fgsm_examples.png`), but accuracy drops by more than half.

### Attack 2: Label Poisoning (OWASP ML02 - Data Poisoning)

**Why we chose it:** FGSM attacks the model after training, so we wanted a second attack on the training data instead. Flipping labels is the simplest kind of poisoning and is realistic when data is crowd-labelled or downloaded from somewhere you don't control. We picked 3 → 8 because the two digits already look similar, so the mistakes are less likely to be noticed.

We changed 20% of the training images labelled 3 (1226 images) to 8 and trained a new model.

| Metric | Baseline | Poisoned model |
|---|---:|---:|
| Clean-test accuracy | 98.88% | 98.86% |
| Clean-test loss | 0.0323 | 0.0553 |
| Actual 3s predicted as 8 | 2 | 11 |
| % of actual 3s predicted as 8 | 0.20% | 1.09% |

Overall accuracy barely moved (0.02%), so someone looking only at accuracy would probably miss the attack. But 3s were predicted as 8 more than five times as often. This is what makes poisoning dangerous. The full confusion matrix is in `reports/run_log.txt`.

### Defense 1: Adversarial Training

Every training batch is used twice: once clean and once as FGSM images (epsilon 0.2).

| Metric | Baseline | Defended model |
|---|---:|---:|
| Clean-test accuracy | 98.88% | 98.05% |
| Clean-test loss | 0.0323 | 0.0563 |
| FGSM accuracy, epsilon 0.05 | 95.39% | 96.23% |
| FGSM accuracy, epsilon 0.10 | 85.44% | 96.13% |
| FGSM accuracy, epsilon 0.20 | 40.18% | 98.27% |

Clean accuracy dropped by 0.83%, and FGSM mostly stopped working against the new model.

One odd thing: the defended model does *better* on FGSM images at epsilon 0.2 than on clean images, and better than at the weaker epsilons 0.05 and 0.1. Epsilon 0.2 is the value it was trained on. This is a known problem with training on FGSM alone (called "label leaking"). The model learns to recognise the FGSM pattern instead of becoming truly robust. So these numbers show that FGSM no longer works well, not that the model is safe against every attack. A stronger attack like PGD would be a better test.

### Defense 2: Training-Label Integrity Check

`data_integrity.py` stores a SHA-256 hash of the training labels in `trusted_labels.sha256`. `baseline.py` and `adversarial_train.py` check the labels against it before training, and stop if they don't match.

Running `data_integrity.py` applies the same 3 → 8 poisoning to a copy of the labels. The hash changes, so the poisoning is detected.

This only works if the hash was created from clean data and is stored somewhere an attacker can't change it. It also only covers the labels, not the images.

## 5. SAST

- **Bandit:** 0 issues in 374 lines of code. We load models with `torch.load(..., weights_only=True)`. Without that, Bandit reports B614 (unsafe PyTorch load, Medium severity), because a malicious `.pth` file could run code when it is loaded.
- **pip-audit:** our first `requirements.txt` was a full `pip freeze` of a shared Python environment (233 packages). pip-audit found 303 known vulnerabilities in 33 of them, including the torch version listed. We cut the file down to the 5 packages the project actually uses, on current versions, and pip-audit now finds nothing.

## 6. Threat Model

See `reports/threat_model.md` for STRIDE, the OWASP ML Top 10 mapping, the attacker model and the LINDDUN notes.

## 7. Conclusion

- A CNN can score almost 99% on clean data and still be easy to fool. FGSM at epsilon 0.2 brought it down to 40.18%.
- Poisoning hardly changed overall accuracy but clearly hurt one class. Accuracy alone isn't enough to catch it; you have to look at the confusion matrix or protect the data.
- Adversarial training mostly stopped FGSM for a small cost (0.83% clean accuracy), but the results suggest it learned FGSM specifically, so it should be tested with stronger attacks.
- A simple checksum can block label tampering before training, as long as the checksum itself is protected.
- Bandit found nothing in our code. The real SAST finding came from the dependencies, which is easy to miss.

## 8. Limitations

- MNIST and a small CNN only. Results on real data would be different.
- FGSM is white-box (it needs the model's gradients).
- The defense was only tested against FGSM.
- The checksum covers labels, not images.
- We didn't test privacy attacks (model inversion, membership inference).

## 9. References

- PyTorch: https://pytorch.org/
- OWASP Machine Learning Security Top 10: https://owasp.github.io/www-project-machine-learning-security-top-10/
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM), 2015: https://arxiv.org/abs/1412.6572
- Kurakin et al., "Adversarial Machine Learning at Scale" (label leaking), 2017: https://arxiv.org/abs/1611.01236
- LINDDUN: https://linddun.org/
- Bandit: https://bandit.readthedocs.io/
- pip-audit: https://github.com/pypa/pip-audit
