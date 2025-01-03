cwd=$(pwd)
python $1/code/preprocess.py $2
python $1/code/smi2desc_nah.py $1/Sample_Input.csv
mv $cwd/Descriptors_Output.csv $1/Descriptors_Output.csv
python $1/code/clean_descriptors.py
python $1/code/desc2model.py $1/Descriptors_Output.csv $1/../checkpoints/normalization.pkl $1/../checkpoints/model.pkl
mv $cwd/Prediction_Results.csv $1/Prediction_Results.csv
python $1/code/postprocess.py $3
rm $1/Sample_Input.csv $1/Descriptors_Output.csv $1/Prediction_Results.csv