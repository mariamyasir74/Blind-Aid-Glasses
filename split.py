import splitfolders
input_folder = "dataset_multiclass_unsplit"
output_folder = "dataset_multiclass"
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.9, 0.1))