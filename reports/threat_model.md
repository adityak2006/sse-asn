# Threat Model

## 1. System

A CNN built with PyTorch takes a 28x28 grayscale image and predicts a digit from 0 to 9.

Parts of the system:
- MNIST training and test data (downloaded by torchvision)
- Training scripts and the saved model weights (`.pth` files)
- The trained model making predictions
- Third-party libraries (PyTorch, torchvision, etc.)

Right now everything runs locally. For the threat model we also think about a real version, where users upload handwriting through a web app and the model is retrained on new data from time to time.

## 2. Attacker Model

**Attacker 1: adversarial input (tested with FGSM)**
- Can send images to the model.
- In our experiment they have white-box access, meaning they know the model and can compute gradients. This is the worst case. A real attacker might only see predictions.
- Goal: make the model give wrong answers with changes too small for a person to notice.

**Attacker 2: data poisoning (tested with label flipping)**
- Can change some training labels, for example a dishonest labeller or a compromised data source.
- Cannot change the model weights or training code directly.
- Goal: make the model make a specific mistake (3 → 8) without lowering overall accuracy enough to be noticed.

**Not considered:** an attacker with full access to the machine. They could simply replace the model file, and no ML defense would stop that.

## 3. STRIDE

| STRIDE | Threat in our system | Control |
|---|---|---|
| Spoofing | Someone pretends to be a trusted data contributor or admin and uploads data or a model. | Authentication, only accept data from known sources. |
| Tampering | Changing input images (FGSM), training labels (poisoning) or the `.pth` model file. | Adversarial training, label checksum, `weights_only=True` when loading models. |
| Repudiation | Someone changes the training data and later denies it. | Log who changed the data and keep versioned datasets and models. |
| Information Disclosure | The model leaks information about its training data (model inversion, membership inference). | Limit access to the model and only return the predicted label, not the confidence scores. |
| Denial of Service | Flooding a prediction API with requests, or sending huge images. | Rate limiting, input size checks. |
| Elevation of Privilege | Loading a malicious pickle `.pth` file runs code with the same permissions as the training script. | `weights_only=True`, run training with limited permissions. |

Tampering is the most important one for this project, and both of our attacks fall under it.

## 4. OWASP ML Top 10 (2023)

| Risk | Applies to us? | Status |
|---|---|---|
| ML01 Input Manipulation | Yes, FGSM | **Tested** + defended (adversarial training) |
| ML02 Data Poisoning | Yes, label flipping | **Tested** + defended (label checksum) |
| ML03 Model Inversion | A deployed model could leak what training digits look like | Not tested |
| ML04 Membership Inference | Could reveal whether someone's handwriting was in the training set | Not tested |
| ML05 Model Theft | Someone could copy the model by querying an API many times | Not tested |
| ML06 AI Supply Chain | Vulnerable or malicious libraries and model files | Checked with pip-audit + Bandit B614 |
| ML07 Transfer Learning | We train from scratch, no pretrained model | Not applicable |
| ML08 Model Skewing | Retraining on biased new data could shift predictions | Not tested |
| ML09 Output Integrity | Predictions changed after leaving the model | Not tested |
| ML10 Model Poisoning | Directly editing the weights in the `.pth` file | Not tested |

## 5. LINDDUN (privacy, for a real version)

MNIST is public and anonymous, so the current project has no real privacy risk. A real handwriting app would collect people's writing, which can be personal (signatures, names, notes).

| LINDDUN | Threat | Mitigation |
|---|---|---|
| Linking | Several uploads can be linked to the same person. | Don't store user IDs with images unless needed. |
| Identifying | Handwriting style can identify a person. | Delete images after prediction, or keep only what is needed. |
| Non-repudiation | Logs prove a user uploaded a certain image. | Keep logs short-term and limit who can read them. |
| Detecting | Someone can tell that a person used the service. | Don't expose activity data. |
| Data Disclosure | Stored images or the model (ML03/ML04) leak data. | Encrypt stored data and restrict model access. |
| Unawareness | Users don't know their writing is used to retrain the model. | Tell them clearly and let them opt out. |
| Non-compliance | Keeping data longer than allowed (e.g. under GDPR). | Retention limits and a way to delete data. |

## 6. Controls Implemented

- **Adversarial training** against ML01. FGSM accuracy at epsilon 0.2 went from 40.18% to 98.27%, and clean accuracy went from 98.88% to 98.05%. The defended model scores higher on FGSM images at epsilon 0.2 than on clean ones, which suggests it learned FGSM specifically, so it should also be tested with stronger attacks.
- **Label checksum** against ML02. Training stops if the labels don't match the trusted SHA-256 hash. It detected our 3 → 8 poisoning.
- **Safe model loading** (`weights_only=True`) against malicious model files.
- **Smaller dependency list**, which removed 303 known vulnerabilities reported by pip-audit (ML06).
