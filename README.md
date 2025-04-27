# Classification and Captioning Aircraft Damage Using Pretrained Models

This project demonstrates the use of pretrained models for two tasks: **image classification** and **image captioning/summarization**. The project is divided into two parts:

1. **Classification of Aircraft Damage** using the VGG16 pretrained model.
2. **Captioning and Summarization of Aircraft Damage Images** using the BLIP pretrained model.

---

## Project Objectives

By completing this project, you will be able to:

- Use the VGG16 model for image classification.
- Prepare and preprocess image data for machine learning tasks.
- Evaluate the model’s performance using appropriate metrics.
- Visualize model predictions on test data.
- Use a custom Keras layer for advanced tasks.
- Generate captions and summaries for images using the BLIP pretrained model.

---

## Key Features

### Part 1: Image Classification
- **Dataset Preparation**: The dataset is automatically downloaded and extracted into training, validation, and testing splits.
- **Data Preprocessing**: Image data is preprocessed using Keras' `ImageDataGenerator` for real-time data augmentation.
- **Model Architecture**: The VGG16 model is used as a feature extractor, with custom dense layers added for binary classification of "dent" and "crack" damage types.
- **Training and Evaluation**: The model is trained and evaluated, with accuracy and loss curves plotted for analysis.
- **Visualization**: Predictions are visualized with true and predicted labels.

### Part 2: Image Captioning and Summarization
- **BLIP Model Integration**: The BLIP pretrained model is used for generating captions and summaries for images.
- **Custom Keras Layer**: A custom Keras layer wraps the BLIP model, allowing seamless integration into TensorFlow workflows.
- **Text Generation**: Captions and summaries are generated based on the content of the images.

---

## Folder Structure

The dataset is organized as follows:

```
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

---

## How to Run the Project

1. **Setup Environment**:
   - Install the required Python libraries using `pip install -r requirements.txt`.
   - Ensure TensorFlow, Keras, and PyTorch are installed.

2. **Run the Notebook**:
   - Open the Jupyter Notebook file: `Classification_and_Captioning_Aircraft_Damage_Using_Pretrained_Models.ipynb`.
   - Execute the cells step-by-step to train the model and generate captions/summaries.

3. **Dataset**:
   - The dataset is automatically downloaded and extracted when the notebook is executed.

4. **Generate Captions and Summaries**:
   - Use the BLIP model to generate captions and summaries for test images.

---

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- PyTorch
- Transformers library
- Matplotlib
- NumPy
- PIL (Pillow)

---

## Results

### Classification
- The VGG16 model achieves high accuracy in classifying "dent" and "crack" damage types.
- Loss and accuracy curves are plotted for training and validation sets.

### Captioning and Summarization
- The BLIP model generates human-readable captions and summaries for aircraft damage images.

---

## Example Outputs

### Classification
- **True Label**: Dent
- **Predicted Label**: Dent

### Captioning
- **Caption**: "This is a picture of a damaged aircraft with visible dents."

### Summarization
- **Summary**: "This is a detailed photo showing significant dents on the aircraft surface."

---

## Acknowledgments

- **VGG16**: Pretrained model for image classification.
- **BLIP**: Pretrained model for image captioning and summarization.
- **Dataset**: Aircraft damage dataset provided for educational purposes.

---

## License

This project is for educational purposes only. Please ensure proper attribution when using the code or dataset.
