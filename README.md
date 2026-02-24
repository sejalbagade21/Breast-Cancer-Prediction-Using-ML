Land Use Classification and Sustainability Trend Analysis

A Deep Learning Approach using Transfer Learning (MobileNetV2)

1-Motivation
Urban expansion is one of the most visible indicators of human impact on the environment.
As cities grow, agricultural land and forest cover often decline — affecting biodiversity, food security, and climate stability.
This project explores how deep learning can assist in monitoring land-use patterns using satellite imagery, and how such classifications can be extended toward sustainability analysis and future trend forecasting.
Rather than treating land classification as a standalone computer vision problem, this work connects it to a broader environmental narrative.

2-Objectives
Develop a robust deep learning model to classify satellite images into major land-use categories.
Analyze the distribution of green cover vs urban land.
Simulate temporal land-use trends.
Forecast potential future land distribution using regression modeling.

3-Dataset Construction
The original dataset consisted of multiple fine-grained land categories.
To align with sustainability-focused analysis, these were consolidated into three macro-classes:

Final Class	Merged Categories
Agriculture	AnnualCrop, PermanentCrop, Pasture, HerbaceousVegetation
Forest	Forest
Urban	Residential, Industrial, Highway

To prevent bias, class balancing was applied (maximum 3000 images per class).
This restructuring enabled a more interpretable environmental comparison between:

🌱 Green Cover (Agriculture + Forest)
🏙 Urban Land

4-Methodology
1️⃣ Data Preprocessing
Image resizing to 160×160
Pixel normalization (rescaling to [0,1])
Data augmentation:Rotation,Zoom,Horizontal flip
80–20 train-validation split
Data augmentation was implemented to improve generalization and reduce overfitting.

2️⃣ Model Architecture
The model uses Transfer Learning with MobileNetV2 (pre-trained on ImageNet).
Why MobileNetV2?
Lightweight and computationally efficient
Strong performance on image classification tasks
Uses depthwise separable convolutions
Suitable for scalable and real-world applications

Architecture:
MobileNetV2 (feature extractor)
GlobalAveragePooling2D
Dense (128 units, ReLU)
Dropout (0.3)
Dense (3 units, Softmax)

3️⃣ Training Strategy

Phase 1 – Feature Extraction
Base model frozen
Custom classifier trained

Phase 2 – Fine-Tuning
Last 30 layers unfrozen
Learning rate reduced (1e-5)

Improved adaptability to land-use features
This two-stage approach allowed leveraging pre-trained features while adapting the network to domain-specific patterns.

5-Evaluation

Model performance was evaluated using:
Accuracy
Training & validation loss curves
Confusion Matrix
Class distribution analysis
The confusion matrix provided insight into inter-class misclassification patterns (e.g., Agriculture vs Forest similarity).

6-Sustainability-Oriented Analysis

After classification, predicted outputs were aggregated to derive:
Green Cover = Agriculture + Forest
Comparative visualizations included:
Land-use distribution pie chart
Green vs Urban bar comparison
Simulated land-use trend graph (2018–2022)

This step bridges computer vision outputs with environmental interpretation.

7-Forecasting Future Trends
To explore potential land-use evolution, simulated trend data was modeled using Linear Regression to predict land distribution in 2025.

⚠ Note:
The temporal data used for forecasting is simulated for demonstration purposes and does not represent real satellite time-series data.
This extension illustrates how classification outputs can feed into predictive sustainability modeling.

8-Technical Stack

Python
TensorFlow / Keras
MobileNetV2 (Transfer Learning)
Scikit-learn
NumPy
Pandas
Matplotlib
Seaborn

9-Key Contributions

Conversion of multi-class satellite dataset into sustainability-focused macro-classes
Implementation of transfer learning with fine-tuning
Balanced dataset preprocessing
Integration of deep learning classification with environmental analytics
Demonstration of trend forecasting from classification outputs

10-Limitations

Forecasting based on simulated trend data
No real temporal satellite sequence
Image classification (not pixel-level segmentation)
Limited geographic contextualization

11- Future Scope

Integration with real multi-year satellite datasets
Use of semantic segmentation (U-Net, DeepLab)
Time-series modeling (LSTM / Prophet)
GIS integration
Deployment as decision-support tool for urban planners

🧩 Broader Impact

This project demonstrates how AI can extend beyond accuracy metrics and contribute to:
Sustainable development research
Urban expansion monitoring
Environmental policy modeling
Climate awareness initiatives
Deep learning models, when contextualized properly, can serve as powerful tools in environmental intelligence systems.
