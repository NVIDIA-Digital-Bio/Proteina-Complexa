# Explainability Subcard

Field | Response
:--- | :---
Intended Task/Domain: | Structural biology
Model Type: | Transformer
Intended Users: | Protein Designers, Structural Biologists, Bioengineers, Computational Biologists, Protein Engineers.
Output: | Types: Text (generated atomistic coordinates and sequence). Formats: Text: PDB file (generated protein with sequence and all atom coordinates)
Describe how the model works: | Complexa uses three neural networks: an encoder, a decoder, and a denoiser, all of which share a core non-equivariant transformer architecture with pair-biased attention mechanisms. For refining the pair representation, optional triangle multiplicative layers can be included within the denoiser network. The architecture operates on a partially latent representation, explicitly modeling the protein's three-dimensional alpha-carbon coordinates while capturing the sequence and all other atomistic details in per-residue eight-dimensional latent variables. The denoiser network parameterizes the flow that maps a noise distribution to the joint distribution of alpha-carbon coordinates and latent variables, which are iteratively updated during the generation process. The decoder then generates the final fully atomistic structure from these outputs. Parameters: 1.6 x 10^8
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable
Technical Limitations & Mitigation: | The primary technical limitation of the Complexa model is its focus on fixed target structures. The current version of the model has not been applied to binder design with flexible targets or to DNA/RNA targets, which is a critical task for many biological applications such as designing de novo binders and enzymes.
Verified to have met prescribed NVIDIA quality standards: | Yes
Performance Metrics: | Co-designability, Diversity, Novelty, Binder Success Metrics like ipAE (interface predicted aligned error) / min ipAE scores.
Potential Known Risks: | When the model does not work as intended, it may produce proteins with poor co-designability (an incompatible sequence and structure) or fail on conditional tasks like binder design by not accurately designing a functional interface. All generated outputs should be treated as proposals that require screening and validation with standard structural assessment tools before any experimental use.
Licensing: | GOVERNING TERMS: Use of the model is governed by the [Licenses](../../licenses/).
