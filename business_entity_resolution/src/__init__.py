"""
business_entity_resolution — Amazon ML Hackathon Solution Package.

Sub-modules
-----------
config          : central configuration (paths, hyper-parameters)
data_loader     : load source TSV files
preprocessing   : text normalisation
blocking        : candidate-pair generation
pair_builder    : assemble labelled/unlabelled pairs
features        : pairwise feature calculation
model           : matching model
evaluation      : precision / recall / F0.5
threshold       : threshold selection
prediction      : test-set prediction pipeline
output          : write TSV submission files
main            : top-level entry point
"""
