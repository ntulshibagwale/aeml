waves
Contains signal processing and machine learning functions
for acoustic emission waves.

docs
Contains project updates and paper documents

experiment_data
Contains raw experimental data, along with filtered data.
Included is a script which is used to generated the filtered
data .json files.

fea
Contains results from ABAQUS simulations.

figures
Contains figures and notebooks used to generate figures.

ml_experiments
Where datasets and logged runs are stored.
Contains datasets used for machine learning experiments,
which are created from the filtered experiment_data
and filtered into training, test, validation.
The logged runs are uploaded to wandb. These experiments
are carried out using ml_pipeline notebook.