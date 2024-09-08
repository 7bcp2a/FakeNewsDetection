# Advanced Machine Learning for Fake News Detection: Classifiers and Neural Networks

## Project Overview
This project, **Advanced Machine Learning for Fake News Detection**, explores various machine learning techniques — both traditional classifiers and neural networks — on detecting fake news. The primary dataset I used is **FakeNewsNet**, which contains two major subsets, **PolitiFact** and **GossipCop**. The project compares different machine learning models, including logistic regression, SVM, Naive Bayes, and CNN, and provides an interactive user interface for real-world fake news detection through a Google Colab notebook.

## Motivation
The rapid spread of fake news, especially during critical events like the COVID-19 pandemic, presents the need for reliable detection systems. This project offers a combination of traditional and neural network approaches and aims to improve accuracy and adaptability in fake news detection.

## Dataset
The project is built on the **FakeNewsNet** dataset, which contains a mixture of news articles, social media interactions, and spatiotemporal data. The dataset includes both true and false news, and is divided into **PolitiFact** and **GossipCop** subsets.

## Models and Approach
The project explores both traditional machine learning classifiers and more advanced neural networks:

- **Classifiers**: Logistic Regression, Support Vector Machine (SVM), and Naive Bayes, applied with techniques such as TF-IDF for feature extraction.
- **Neural Networks**: Convolutional Neural Networks (CNNs) are implemented to capture more complex patterns in the textual data.
- **Optional Models**: Bi-GCN (Graph Convolutional Networks) and Recursive Neural Networks (RvNNs) are planned for future implementation.

## Notebooks and Workflow
1. **Preprocessing**: Prepares the FakeNewsNet dataset by cleaning and normalizing data, it also handles missing values, text normalization, and stemming.
2. **Classifier Models**: Builds logistic regression, SVM, and Naive Bayes classifiers, evaluates their performance using accuracy, precision, recall, F1-score, and ROC-AUC.
3. **CNN Models**: Implements CNNs using TensorFlow for detecting fake news. Performance is improved through hyperparameter tuning and k-fold cross-validation.
4. **Inference**: A user-friendly Google Colab notebook allows users to input news headlines and receive real-time predictions on whether the news is real or fake. The models used in this notebook are the best-performing CNNs from the evaluation process.

## Results
- **Classifiers**: The SVM model with hyperparameter tuning performed the best among traditional classifiers.
- **Neural Networks**: CNN models outperformed traditional classifiers on both PolitiFact and GossipCop datasets. This displayed their ability to capture patterns in the data.

## Installation and Setup
Clone the repository and open the provided Jupyter Notebooks in Google Colab for execution. 

### Prerequisites
- Python 3.x
- Libraries: TensorFlow, Scikit-learn, Pandas, NumPy, IPython Widgets (ipywidgets)

### Installation
Install the required dependencies:
```
pip install tensorflow scikit-learn pandas numpy ipywidgets
```

## Usage

- **Preprocessing Notebook**: Run this notebook first to clean and prepare the dataset.
- **Classifier and CNN Notebooks**: Train models using the classifiers or CNN notebooks and evaluate the performance of each model.
- **Inference Notebook**: Load the inference notebook in Google Colab and input news titles to detect fake news using the trained models.

## Performance Evaluation

All models were tested using metrics like accuracy, precision, recall, and F1-score. Cross-validation and hyperparameter tuning were performed to make sure the results were robust. Detailed performance analysis is included in the report.

## Future Work

- Implement Bi-GCN and RvNN models for graph-based learning on the FakeNewsNet dataset.
- Explore unsupervised and semi-supervised learning techniques to reduce reliance on labeled datasets.
- Expand the dataset to include non-English content and improve cultural and linguistic diversity.

## License

This project is licensed under the MIT License.

## Acknowledgements

The project is based on the CM3060 Natural Language Processing template for Fake News Detection from the University of London. I would like to send a special thanks to the developers of the FakeNewsNet dataset and the research community for contributions in the field of fake news detection.
