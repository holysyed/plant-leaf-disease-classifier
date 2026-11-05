# Plant Leaf Disease Classifier

A beginner-friendly AI/ML project that classifies plant leaf images into disease categories with transfer learning using MobileNetV2. It includes a training script that learns from labeled image folders, a command-line predictor, and a Streamlit web demo for uploading leaf photos. The project works with PlantVillage or another dataset organized into one folder per class. It is intended for learning and demonstration; predictions are model estimates and may not generalize to real field conditions.

## 1. Set up the dataset

Download and extract the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset). Put the class folders under `data/PlantVillage/`, like this:

```text
data/PlantVillage/
├── Tomato___Early_blight/
│   ├── image_1.jpg
│   └── ...
├── Tomato___Late_blight/
│   └── ...
└── Tomato___healthy/
    └── ...
```

The training script uses every class folder it finds. To keep the first experiment manageable, you can start with two or three tomato classes. Use the same folder layout if you choose another image dataset.

## 2. Install dependencies

Use Python 3.10–3.12 and run these commands from the project folder:

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install -r requirements.txt
```

## 3. Train

```bash
python train.py --data-dir data/PlantVillage --epochs 10
```

The script reserves 20% of the images for validation and writes `artifacts/plant_disease_model.keras` and `artifacts/class_names.json`. The default ImageNet weights are downloaded the first time; use `--weights none` to train without downloading them (this usually needs more data and training).

## 4. Make a prediction

```bash
python predict.py path/to/leaf.jpg
```

## 5. Launch the demo

```bash
streamlit run app.py
```

## Good next steps for a class presentation

- Record training and validation accuracy and loss for each epoch.
- Compare the pretrained model with `--weights none`.
- Review mistakes from each class and explain likely causes.
- Try photos taken outside the dataset and discuss why performance may change.

PlantVillage images are captured in controlled conditions. A high validation score on those images does not prove the model will work reliably on real crops, and predictions should not be treated as a diagnosis.
