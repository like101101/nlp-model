# nlp-model

Template for Using natural language processing to make prediction

## Step 1: Place your csv data

After you have cloned this repo, inside the working directory, do

`mkdir data`

Then place your csv data under this data directory

## Step 2: Modify the column name

open the two python file, make sure that the column name in the csv file is same as in dataframe
if not, please modify it before use

## Step 3: Running the model

run the preprocessor by using

`python3 preprocess.py`

Note that this might take up to hours depending on your csv file size

then run the model fitting by using

`python3 model.py`

Your will see time it takes to run the model as well as the accuracy score.
