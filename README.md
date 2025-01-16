# Internationalizing Domain Generation Algorithm (DGA) Detection

This repository contains the source code for the internationalization of the DGAs
*corebot*, *dircrypt*, *qadars*, *pushdo*, *charbot*, *gozi*, *nymaim2*, *suppobox*, *orchard* and
*sharkbot*. The original implementations of these DGAs originate from
[Johannes Bader](https://github.com/baderj/domain_generation_algorithms).

Furthermore, this repository contains the source code for the corresponding experiments:

- Include-only-one-language
- Leave-one-language-out
- Include-only-one-script
- Leave-one-script-out
- Include-only-one-IDGA
- Leave-one-IDGA-out
- Include-only-one-IDGA-variant
- Leave-one-IDGA-variant-out

Additionally, the source code contains the training details and architectural backbone
of DomainBERT, our [DeBERTaV3](https://github.com/microsoft/DeBERTa)-based DGA classifier.
The fine-tuning procedure heavily utilizes the training procedure implemented in
the [transformers](https://github.com/huggingface/transformers) library.

## Structure

| Folder | Description                                                                                                                                               |
|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| idga   | Implementation of internationalized DGAs (IDGA)                                                                                                           | 
| dga_classifier | Implementation of DGA classification training                                                                                                             |

Checkout the READMEs in the corresponding folders.

## Installation

This repository uses `poetry` as its package manager. Use `poetry install --no-root` to setup the necessary Python environment.

