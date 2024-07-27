# Natural Language Inferencing (NLI) Model

## Project Overview

For any two sentences, there are three possible relationships:
1. **Entailment**: One sentence logically follows from the other.
2. **Contradiction**: One sentence logically contradicts the other.
3. **Neutral**: The sentences are unrelated.

Natural Language Inferencing (NLI) is a popular NLP problem that involves determining how pairs of sentences (consisting of a premise and a hypothesis) are related. Your task is to create an NLI model that assigns labels of `0`, `1`, or `2` (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses.

## Setup Instructions

### Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kyoos15/NLI_Project.git
    cd NLI_Project
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data

Place your training, validation, and test datasets in the `data/` directory. Ensure the datasets have the following columns:
- `premise`: The premise sentence.
- `hypothesis`: The hypothesis sentence.
- `label`: The label (0 for entailment, 1 for neutral, 2 for contradiction).

## Training the Model

We trained the model in the following way:

Loads and preprocesses the data.
Tokenizes the data using the BERT tokenizer.
Initializes the BERT model for sequence classification.
Trains the model with the specified parameters on the SNLI dataset.
Saves the best model based on validation accuracy.
Visualises the model's prediction through confusion matrix and classification report.

## Key Features

- **Data Handling**: Efficient loading and preprocessing of large datasets.
- **BERT Tokenization**: Utilizes the BERT tokenizer for processing sentences.
- **Custom Dataset Class**: Implements a PyTorch `Dataset` class for handling tokenized data and labels.
- **Model Training**: Uses BERT for sequence classification, with options for early stopping and learning rate scheduling.
- **Evaluation**: Provides detailed evaluation metrics including accuracy, confusion matrix, and classification report.
- **Visualization**: Generates visualizations for data exploration and model performance.
- **GPU Support**: Leverages GPU for faster training and inference if available.

# Contributors
* Vaibhav Ojha
* Anjanayae Chaurasia
* Kushagra Shrikhande
* Harsh Anand

# Mentor
* Anushka Jha
