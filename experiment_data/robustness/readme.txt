01_experiment_data folder

contains all the raw AE data collected from the experiments.
Each experiment folder will contain the raw waveforms
and times file collected from the experiment, generated
by the Digital Wave software.

The create_filtered_dataset file is run on an experimental
folder to clean the dataset and associate the meta data
from the file names with the waveform. The output of this
python script is the filtered_json file which goes into
filtered folder.