# LION-BO ReadMe

The code is based on Josie's Bayesian Optimisation for quad positioning at SCAPA, which is where the name comes from.

## Main Files

The main two files here are:
- `LION_EnvelopeBayesianAnalysis.py`
- `LION_EnvOptimisation.py`

## Running the Code

Run using:
```bash
python ../LION-BO/LION_EnvelopeBayesianAnalysis.py -i source.dat -n 100000 -o test.dat
```

The output file I do not believe is properly used at this moment.

The code is currently setup for just running the cost function once and returning a histogram using the response functions Nick gave. This is to verify whether the shifts and tilts are creating the expected effect on the beam. This is why it will plot out the histograms of each layer, which obviously it won't when optimising. These should be outputted to 99-Scratch as:

```
99-Scratch/sim_data_histogram_layer_{i+1}.png
```

The experimental data and response functions should be in the folder. The experimental data has just been made so apologies if it fails. It is named as `Film_{min_film_energy}MeV_59x59.csv`. Apologies for the hard coding nature of the response functions location but they will get removed once G4 sim is included.

The `RCF_Centering.ipynb` is the notebook I use to do the RCF alignment. Feel free to have a look at that too.

The `Wider_Cropped_RCF_Data` is the energy deposition data from Nick before alignment.

`checking_response_func.py` was used to verify the response functions looked as expected.
