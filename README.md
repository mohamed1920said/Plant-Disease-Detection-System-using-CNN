# Plant Disease Detection Using a CNN

This repository is an educational image-classification project for identifying plant leaf conditions. It contains a TensorFlow/Keras training notebook, a class-activation-map notebook, two Streamlit prototypes, saved training history, and a project report.

The repository is not a ready-to-run packaged application: the dataset and trained model are not committed, and several notebook and application paths point to the original author's Windows machine. Follow the preparation notes below before running it.

## Project workflow

1. Download and arrange the plant-disease image dataset.
2. Update the dataset paths in `Train_plant_disease.ipynb`.
3. Train the 38-class CNN and save `trained_model.keras`.
4. Update the model path in the selected Streamlit application.
5. Run the application and upload a leaf image for classification.
6. Optionally use `CAM.ipynb` to explore model attention.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Train_plant_disease.ipynb` | Dataset loading, CNN definition, training, evaluation, plots, and model export |
| `CAM.ipynb` | Experimental class-activation-map analysis |
| `plant_disease + sol.py` | Main Streamlit prototype with the complete 38-label list and general treatment text |
| `test1.py` | Earlier Streamlit prototype with only eight labels |
| `training_hist.json` | Recorded loss and accuracy values for a 10-epoch training run |
| `requirement.txt` | Pinned Python dependencies used by the project |
| `test/test_main.py` | Placeholder smoke test; it does not test the model or application |
| `Rapport de Projet IA mohamed said.docx` | Project report |

## Model and data

The training notebook references the Kaggle **New Plant Diseases Dataset (Augmented)** and expects separate `train` and `valid` directories. The dataset itself is not included in this repository. Confirm the dataset's license and usage terms at its source before downloading or redistributing it.

The checked-in notebook builds a sequential CNN with:

- 128 x 128 RGB input images;
- five convolution/max-pooling stages using 32, 64, 128, 256, and 512 filters;
- a 1,500-unit dense layer;
- a 38-unit softmax output;
- Adam with a learning rate of `0.0001`;
- categorical cross-entropy; and
- 10 training epochs.

The class labels cover healthy and diseased leaves from apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, and tomato plants. Class order must remain identical between training and inference. Validate the order from `training_set.class_names` before using a newly trained model.

## Recorded results

The values below come from the committed `training_hist.json`, not from a fresh reproduction:

| Metric | Recorded value |
| --- | ---: |
| Final training accuracy, epoch 10 | 98.21% |
| Final validation accuracy, epoch 10 | 95.67% |
| Best recorded validation accuracy | 97.14% at epoch 8 |
| Final training loss | 0.0549 |
| Final validation loss | 0.1458 |

Some application/report text mentions approximately 99% or higher accuracy. That claim does not match the committed history above and should not be presented as reproduced without the exact model, dataset split, and a repeatable evaluation. The repository also has no independent test-set metrics or confusion-matrix artifact that establishes field performance.

## Setup

TensorFlow 2.10 is an older pinned dependency, so use a Python version and platform supported by that TensorFlow release. A Python 3.9 or 3.10 virtual environment is a practical starting point.

The checked-in dependency set is old and has not been reproduced as part of this documentation review; `streamlit` is also left unpinned. Test installation in a clean environment, record the resolved versions, and document any changes used to reproduce training.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirement.txt
pip install jupyter
```

### Linux or macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirement.txt
pip install jupyter
```

If TensorFlow 2.10 is unavailable for your operating system or Python build, create a compatible environment rather than silently changing versions. A dependency upgrade may require retraining and regression testing.

## Train the model

1. Download the dataset linked in `Train_plant_disease.ipynb`.
2. Open the notebook:

   ```bash
   jupyter lab Train_plant_disease.ipynb
   ```

3. Replace the hard-coded absolute Windows paths for the training, validation, and test images with paths on your machine.
4. Run the notebook from top to bottom.
5. Confirm that the exported model has 38 outputs in the same order used by the application.

The notebook saves `trained_model.keras` in its current working directory. It also writes training history to JSON. Neither a `.keras` nor a `.h5` model is committed in the current repository, so inference cannot run immediately after cloning.

## Run the Streamlit application

The more complete prototype is `plant_disease + sol.py`. Before starting it, replace these original-machine paths inside `model_prediction`:

```python
model_keras_path = r"C:\essat\projet ia\trained_model.keras"
model_h5_path = r"C:\essat\projet ia\trained_model.h5"
```

Point one of them at the model created by the training notebook, then run:

```bash
streamlit run "plant_disease + sol.py"
```

The application resizes uploaded images to 128 x 128 pixels, selects the largest softmax output, displays confidence, and provides general French-language treatment text.

Do not use `test1.py` with a 38-output model without correcting its label mapping. That prototype contains only eight labels, so a prediction index outside that list is invalid and even an in-range index may be misleading if the training class order differs.

## Class-activation-map notebook

`CAM.ipynb` is an exploratory visualization notebook. It also contains absolute paths for:

- `trained_model.h5`; and
- the local training dataset.

Update both paths before running it. Confirm that the selected convolutional layer and preprocessing steps match the model you trained; otherwise, the generated heat map may be invalid or uninformative.

## Current limitations

- The dataset is not included.
- The trained `.keras`/`.h5` model is not included.
- Training, CAM, and application files contain machine-specific absolute paths.
- The two Streamlit scripts use different label lists and different amounts of treatment information.
- The model is loaded again for every prediction instead of once at application startup.
- The displayed environmental values (32 C temperature, 40% humidity, and 30% soil moisture) are fixed demonstration values, not live sensor readings.
- The only automated test checks that `1 + 1 == 2`; model loading, preprocessing, class mapping, and UI behavior are untested.
- No dataset checksum, random seed, complete environment lockfile, or model provenance record is provided.
- No project license file is included.

## Responsible use

This is a learning prototype, not a validated agronomy or crop-protection system. Image predictions can be wrong because of lighting, background, cultivar, camera quality, symptoms shared by different diseases, or data outside the training distribution. Treatment text in the application is general information only. Confirm a diagnosis and any chemical treatment with a qualified agronomist and follow local labels, regulations, protective-equipment requirements, and pre-harvest intervals.

For a reproducible release, add the model's source and checksum, dataset version and split procedure, full evaluation results, supported Python environment, and a clear software/data license.
