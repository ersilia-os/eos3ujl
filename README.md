# Mtb Permeability

This model predicts the probability of a compound of passing the Mtb cell wall membrane. The classifier (permeable vs not permeable) model was trained on a dataset of 5368 molecules. It is a simple classifier (SVC) using Mordred descriptors.

## Identifiers

* EOS model ID: `eos3ujl`
* Slug: `mtb-permeability`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Classification`
* Output: `Probability`
* Output Type: `Float`
* Output Shape: `Single`
* Interpretation: Probability of a compound passing the Mtb cell wall membrane

## References

* [Publication](https://link.springer.com/article/10.1007/s11030-024-10952-3)
* [Source Code](https://github.com/PGlab-NIPER/MTB_Permeability)
* Ersilia contributor: [miquelduranfrigola](https://github.com/miquelduranfrigola)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos3ujl)
* [AWS S3](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos3ujl.zip)
* [DockerHub](https://hub.docker.com/r/ersiliaos/eos3ujl) (AMD64, ARM64)

## Citation

If you use this model, please cite the [original authors](https://link.springer.com/article/10.1007/s11030-024-10952-3) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a GPL-3.0-or-later license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!