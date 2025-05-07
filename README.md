# Mtb cell wall permeability

This model predicts the probability of a compound of passing the Mycobacterium tuberculosis cell wall membrane. The classifier (permeable vs not permeable) model was trained on a dataset of 5368 molecules. It is a simple classifier (SVC) using Mordred descriptors.

This model was incorporated on 2024-10-16.

## Information
### Identifiers
- **Ersilia Identifier:** `eos3ujl`
- **Slug:** `mtb-permeability`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Tuberculosis`
- **Target Organism:** `Mycobacterium tuberculosis`
- **Tags:** `M.tuberculosis`, `Permeability`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability score of a compound passing the Mtb cell wall membrane

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| permeability_probability | float | high | Probability of a compound permeating through the Mycobacterium tuberculosis cell wall |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos3ujl](https://hub.docker.com/r/ersiliaos/eos3ujl)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos3ujl.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos3ujl.zip)

### Resource Consumption
- **Model Size (Mb):** `150`
- **Environment Size (Mb):** `721`
- **Image Size (Mb):** `917.2`


### References
- **Source Code**: [https://github.com/PGlab-NIPER/MTB_Permeability](https://github.com/PGlab-NIPER/MTB_Permeability)
- **Publication**: [https://link.springer.com/article/10.1007/s11030-024-10952-3](https://link.springer.com/article/10.1007/s11030-024-10952-3)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-or-later](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos3ujl
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos3ujl
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
