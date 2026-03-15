# Model Overview

## Model NSpect ID
NSPECT-X5VA-YFLY

### Description
Proteina-Complexa is a state-of-the-art generative model that designs fully atomistic protein structures for binder design to both protein and small molecule targets, generating both the sequence and all atomic coordinates. It is trained using a partially latent flow matching objective, where the protein backbone is modeled explicitly while side-chain details and sequence information are captured in a fixed-size latent space. New proteins are generated iteratively starting from random noise, using stochastic sampling. This framework enables a protein designer to generate novel, fully atomistic protein structures conditioned on target structures to generate binding proteins for diverse use cases. Its ability to perform motif scaffolding at the same time also allows applications in enzyme design.

This model is ready for commercial use.

### License/Terms of Use

Governing Terms: Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### Deployment Geography
Global.

### Use Case
Proteina-Complexa can be used by protein designers interested in generating novel fully atomistic protein binder structures and their corresponding sequences.

### Release Date
**Github** [03/15/2026] via [https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa]
**NGC** [03/16/2026] via [Proteina-Complexa](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa), [Proteina-Complexa-Ligand](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa_ligand), [Proteina-Complexa-AME](https://registry.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa_ame)
**Hugging Face** [03/15/2026] via [Proteina-Complexa](https://huggingface.co/nvidia/NV-Proteina-Complexa-Protein-Target-160M-v1), [Proteina-Complexa-Ligand](https://huggingface.co/nvidia/NV-Proteina-Complexa-Ligand-Target-160M-v1), [Proteina-Complexa-AME](https://huggingface.co/nvidia/NV-Proteina-Complexa-AME-160M-v1)

## Reference(s)
[Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute](https://openreview.net/forum?id=qmCpJtFZra)

## Model Architecture
**Architecture Type:** Autoencoder + Flow model.
**Network Architecture:** Transformer.

Proteina-Complexa uses three neural networks, an encoder, a decoder, and a denoiser, all of which share a core non-equivariant transformer architecture with pair-biased attention mechanisms. For refining the pair representation, optional triangle multiplicative layers can be included within the denoiser network. The architecture operates on a partially latent representation, explicitly modeling the protein's three-dimensional alpha-carbon coordinates while capturing the sequence and all other atomistic details in per-residue eight-dimensional latent variables. The denoiser network parametrizes the flow that maps a noise distribution to the joint distribution of alpha-carbon coordinates and latent variables, which are iteratively updated during the generation process. The denoiser also takes as inputs three-dimensional target structure and sequence for protein or small molecule targets that are then used as conditioning for the design of the binding protein. The decoder then generates the final fully atomistic binder structure from these outputs.

**This model was developed based on:** [La-Proteina](https://github.com/NVIDIA-Digital-Bio/la-proteina).
**Number of model parameters:** 1.6 x 10^8

## Input
**Input Type(s):**
- Text (time step schedules, noise schedules, sampling modes, motif coordinates, target coordinates)
- Number (number of residues, noise scales, time step sizes, seed, noise schedule exponents)
- Binary (use of self-conditioning)

**Input Format:**
- Text: Strings (time step schedules, noise schedules, sampling modes), PDB file (motif coordinates)
- Number: Integers (number of residues, seed), floats (noise scales, time step sizes, noise schedule exponents)
- Binary: Booleans

**Input Parameters:**
- Text: One-dimensional (1D) or text file (PDB file)
- Number: One-dimensional (1D)
- Binary: One-dimensional (1D)

## Output
**Output Type(s):** Text (generated atomistic coordinates and sequence)
**Output Format:** Text: PDB file (generated protein with sequence and all atom coordinates)
**Output Parameters:** One-dimensional (1D)
**Other Properties Related to Output:** The model output is stored as a PDB file containing the protein sequence and three-dimensional coordinates for all atoms.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration
**Runtime Engine(s):** PyTorch
**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere
**Preferred/Supported Operating System(s):** Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s)
We release three generative model checkpoints and their corresponding autoencoders.

Generative model checkpoints:
- `complexa.ckpt` — Design of binders to protein targets. Used with `complexa_ae.ckpt`.
- `complexa_ligand.ckpt` — Design of binders to small molecule targets (LoRA fine-tuned). Used with `complexa_ligand_ae.ckpt`.
- `complexa_ame.ckpt` — Design of binders to small molecule targets with motif scaffolding support for enzyme design (LoRA fine-tuned). Used with `complexa_ame_ae.ckpt`.

Autoencoder checkpoints:
- `complexa_ae.ckpt` — Autoencoder for protein binder model. Trained with proteins up to 256 residues. Finetuned on PDB. 
- `complexa_ligand_ae.ckpt` — Autoencoder for the ligand binder model. Trained with protein binders up to 512 residues. Same AE as LaProteinaAE v1.1 in La-Proteina.
- `complexa_ame_ae.ckpt` — Autoencoder for the AME motif scaffolding model. Trained with protein binders up to 256 residues. Same procedure as LaProteinaAE v1.1 but trained up to length 256.

## Training, Testing, and Evaluation Datasets

### Dataset Overview
**Total Size:** Approximately 1,300,000 data points
**Total Number of Datasets:** 4 (AFDB, Teddymer, PDB and PLINDER)
**Dataset partition:** Training 99.9%, Validation 0.1%
**Time period for training data collection:** The AFDB was generated using AlphaFold 2, published in 2021. The PDB consists of experimental data, which started being collected in 1971.
**Time period for testing data collection:** See above.
**Time period for validation data collection:** See above.

## Training Dataset

**(1) AFDB**
**Link:** https://alphafold.ebi.ac.uk/
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The AlphaFold Protein Structure Database (AFDB) contains approximately 214M synthetic three-dimensional protein structures predicted by AlphaFold2, along with their corresponding sequences. We trained Complexa models on a subset of the AFDB, one comprising 344,508 structures.

**(2) Teddymer**
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The Teddymer dataset contains around 500,000 synthetic dimer pairs created by annotating, filtering and cropping structure predictions from the AFDB.

**(3) PDB**
**Link:** https://www.rcsb.org/
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Experimental structure determination (X-ray crystallography, cryo-EM, NMR)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We use a filtered subset of the PDB during model training.

**(4) PLINDER**
**Link:** https://console.cloud.google.com/storage/browser/plinder
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Experimental structure determination (X-ray crystallography, cryo-EM, NMR)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We use a filtered subset of PLINDER (around 80,000 structures) as training set for our ligand models.

### Testing Dataset

**(1) AFDB**
**Link:** https://alphafold.ebi.ac.uk/
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We use a subset of the AFDB subset of 734,658 structures as a reference set in evaluations. We also extract a subset of files, and modify them to create the benchmark for binder design. We release these modified files with the codebase.

**(2) PDB**
**Link:** https://www.rcsb.org/
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Experimental structure determination (X-ray crystallography, cryo-EM, NMR)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We use the entire PDB as reference set in evaluations. We also extract a subset of files, and modify them to create the benchmark for binder design. We release these modified files with the codebase.

### Evaluation Dataset
**Link:** https://alphafold.ebi.ac.uk/
**Data Modality:** Text (PDB files)
**Non-Audio, Image, Text Training Data Size:** Between 0.5MB and 2MB per sample (PDB file)
**Data Collection Method by dataset:** Synthetic (AlphaFold predictions)
**Labeling Method by dataset:** N/A
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** We use a subset of the AFDB subset of 1,300 structures as a validation set during training.

### Quantitative Evaluation Benchmarks

#### Small molecule targets
Quantitative evaluation of generative performance on four small-molecule binders (SAM, OQO, FAD, IAI) without test-time optimization.

| Model | Time (s) | Novelty | SAM unique successes | OQO unique successes | FAD unique successes | IAI unique successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RFDiffusion-AllAtom | 2 | 0.72 | 2 | 3 | 5 | 8 |
| Proteina-Complexa (ours) | 10 | 0.71 | 10 | 6 | 17 | 19 |

#### Protein targets
Quantitative evaluation of generative performance on protein binder targets without test-time optimization, comparing Complexa to prior generative baselines.

| Model | Time (s) | Novelty | Self unique successes | MPNN-FI unique successes | MPNN unique successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| RFDiffusion | 4.68 | 0.87 | 3 | 70.8 | – |
| Protpardelle-1c | 0.73 | 0.77 | 0 | 8.13 | – |
| APM | 0.31 | 0.86 | 1 | 73.1 | 3.15 |
| Proteina-Complexa (ours) | 9.10 | 0.80 | 14 | 15.6 | 14.4 |

#### Inference-time optimization
Quantitative evaluation of beam-search-based test-time optimization with different folding and hydrogen-bond rewards.

| Configuration | Unique successes (avg.) | Interface H-bonds (avg.) |
| --- | ---: | ---: |
| Proteina-Complexa, no reward | 77.00 | 5.271 |
| Proteina-Complexa, with f_ipAE | 83.36 | 5.524 |
| Proteina-Complexa, with f_H-Bond | 82.36 | 7.154 |
| Proteina-Complexa, with f_ipAE + f_H-Bond | 86.26 | 6.518 |

For more information: [Paper](https://openreview.net/forum?id=qmCpJtFZra)

## Inference
**Acceleration Engine:** PyTorch
**Test Hardware:** NVIDIA A100

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the [Model Card++ Bias](bias.md), [Explainability](explainability.md), [Safety & Security](safety.md), and [Privacy](privacy.md).

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
