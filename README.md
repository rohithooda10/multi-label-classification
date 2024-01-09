# Multi Label Classification
(Trained models in Documentation)
To run the code, simply open the notebook in Google colab and run all cells.
[NOTE: To load and save the model, you will need to mount your google drive, command for this is added at the top.
Mount your drive and keep the saved model at '/content/drive/My Drive/saved_model.pth'.]
## Introduction

The goal of this project is to develop a multilabel legal text classification model for the European Court of Human Rights (ECHR) dataset. The dataset contains legal documents, and the task is to predict the articles of the European Convention on Human Rights allegedly violated by each case.

### Dataset description
We are using ecthr_b dataset from the Lex GLUE collection, which is designed for the task of classifying legal documents tailored for the European Court of Human Rights (ECHR).
<img width="1267" alt="Screenshot 2024-01-09 at 1 51 57 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/1da0385c-db20-40f7-bdee-1131de9f327a">

- Class distribution
<img width="1010" alt="Screenshot 2024-01-09 at 1 49 37 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/90ced8d6-52aa-4475-a24c-4a5cddc1c9e9">

- Text length distribution
<img width="886" alt="Screenshot 2024-01-09 at 1 49 54 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/8e4a8ce3-7ff5-4f92-bc36-406140794414">

### Features & Labels
In the dataset specified above, we have just two features -
#### text:
This feature consists of textual content of the legal document, which is a list of factual paragraphs (facts) from the case description. Basically the legal document is divided into paragraphs and each string in the list represents a paragraph.
#### labels:
This feature is a list of numbers representing articles of ECHR that were allegedly violated (considered by the court) by the case specified in the document. For example, if the document violates article 3 and 4, labels will be [3, 4].

## Process/ Approach

### Data Loading and preprocessing

#### Removed duplicates and missing data to ensure data quality.
Duplicates Removal: Remove duplicate entries in the dataset based on the 'text' feature.
Missing Data Handling: Drop rows containing missing values after duplicate removal.
Reset Index: Reset the indices after performing operations to maintain indexing.

#### Combined paragraphs to get complete documents for each case.
We concat all strings representing paragraphs of the document to form a complete document. We keep these combined text in column 'text_column'

#### Performed text preprocessing
Lowercasing: Convert all text to lowercase for uniformity going forward.
Removing Numbers: Eliminate numebrs and dates to focus on text.
Removing Non-English Words: Utilize NLTK's stopwords to filter out non-English words and special characters.
Stopword Removal: Eliminate stopwords from the text and keep only meaningful content.

### Tokenization and Dataset Creation

#### Tokenization: 
Utilized the BERT tokenizer from the transformers library to convert text into tokens.
Padding and Truncation: Apply padding and truncation to make sure that all input sequences have the same length, this fixed length is defined as 1500.

#### Custom datasets and dataloader:
Created custom datasets and data loaders for training, validation, and testing to efficiently load batches of data during training and evaluation.

### Model Architecture

#### Custom classifier
We use pretrained BERT on legal cases, but since we want to use BERT for classification, we add our own custom linear layer at the end which gives use result of length equal to number of classes.
Since the sequences are much largers than 512 which BERT allows to give as an input, we run a 512 words sized window on the input sequence and get output for each 512 words. Then we take output of all these windows and take their average.
Then the average is passed on to the classifier layer which gives us logits for the input.

#### Initialization: 
Initialized the model, optimizer, and loss function. And since we saw some classes occured more than others, we gave some extra weightage to the other classes.

### Training

Trained the model over multiple epochs, monitoring training loss, accuracy, and making use of class weights for imbalanced classes.
Validated the model performance on a separate validation set to assess generalization.

### Testing and Evaluation

Tested the trained model on the test set and evaluated performance using metrics such as accuracy, precision, recall, and F1 score. Additionally, eported classification results.

### Inference

Implemented an inference function to predict labels for new legal texts.


## Challenges Faced, Choices and Solutions

### Class Imbalance
Observation: 
During data exploratory part of the process, we observed that in the data some classes occurred a lot more than others. For eaxmple class 3 occurred much more than any other. 

Decision/ Solution: 
To prevent the model to just focus more on that label and just give the most common label as answer, it loss function, we added class weights based on number of occurence of classes. This would prevent model to learn to just push out the most common class as answer everytime. Although the weighted classes works in theory for our particular case, hyperparameters and dataset, it didn't lead to any substaintial result improvement.

### Text Lengths

Observation: 
Document text were huge. Documents ranged from few hundreds to 5000+. This would cause problems with our pretrained model, BERT, since BERT only allows input sequence of max length 512 words. Average length of the text was around 1000 - 1500 after removal of stop words.

Decision/ Solution: 
We had few choices, to use a model that can handle these long sequence such as BigBird or Longformer, but using these models crashed runtime on colab several times suggesting their need for higher computing power.

Then I just use BERT and let it truncate sequences at 512 words. It did give good results, but according to few researches, it was shown that just truncating last part of the sequence may cause loss of lot of information and context, hence I kept searching for better solution.

Then I decided to go with a windowed approach, in this I give input to BERT in 512 words and then move window to next 512 words and do the same. All the outputs obtained from each window are averaged to get single output. This output is passed through classifier layer to get logits. This apporach gave best result. But I had to make some sacrifices due to time and computation limitations and work with max length of 1500. This length covered around 75% of the sequences, which was good enough for our task.

## Some other decisions

### Loss function
I chose BCEWithLogits loss function since our task involves predicting multiple labels for each document (ie. multi-label classification). It independently computes the binary cross-entropy loss for each label, allowing the model to handle multiple labels for a single instance. Additionally, BCEWithLogits loss function supports the use of class weights, which can be used when there is class imbalance

### Optimizer
I choose AdamW because it combines the benefits of momentum like SGD with momentum and adaptive learning rate. This combination enables faster convergence along relevant dimensions and slower updates along noisy dimensions.

## Findings and Results

The model achieved satisfactory performance on the validation and test sets, giving nearly similar performance as given in benchmarks on huggingface demonstrating its ability to classify legal texts into the relevant violated articles of the ECHR.

The preprocessing steps effectively handled noise in the legal texts, and the window-based approach in the model addressed challenges related to long text lengths.

## Compare Pretrained V Finetuned BERT
In finetuned, we finetune both the newly added classifier layer and the BERT weights as well. Here are some results -

- Classification report
<img width="483" alt="Screenshot 2024-01-09 at 11 10 48 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/7f114022-28d0-480a-9cd3-13f4eb8e9872">


- Finetuned accuracies
<img width="699" alt="Screenshot 2024-01-09 at 11 08 43 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/d7ef7b78-8a28-4505-8b65-4d757e70224c">


- Finetuned training loss per epoch
<img width="699" alt="Screenshot 2024-01-09 at 11 08 36 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/b36fbd29-f176-40ca-9522-60730517fa77">


In pretrained BERT, we just a classifier layer and trained that layer only while keeping BERT layers frozen. Here are some results -

- Pretrained Accuracies
<img width="710" alt="pretrained accuracies" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/d12fb62e-80d1-452f-a5db-ea1bb1cf785c">

- Pretrained Training loss per epoch
<img width="710" alt="pretrained train loss" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/72b63ab6-a8a9-486a-aea5-f9e190cd982b">

- Classification report
<img width="481" alt="Screenshot 2024-01-09 at 1 56 23 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/52556719-156d-4a2a-bebb-464a49885ced">

Comparison

|   Metric  | Finetuned | Pretrained |
|-----------|-----------|------------|
|  Accuracy |   0.553   |   0.163    |
| Precision |   0.798   |   0.445    |
|   Recall  |   0.744   |   0.207    |
|  F1 Score |   0.770   |   0.283    |

## Sample inference
<img width="1178" alt="Screenshot 2024-01-09 at 11 21 56 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/3567dea7-1411-44c2-a10c-fcb752a13893">

Inference Output

<img width="218" alt="Screenshot 2024-01-09 at 11 50 22 AM" src="https://github.com/rohithooda10/multi-label-classification/assets/109358642/a30dcd64-0754-4171-b265-3fbafb49a5b6">

## Future scope/ Improvements
Since we ran our model within a computation and time limited environment, we were restricted to only few experimentations. With more powerful machine, we can try out different learning rates, higher epochs, bigger batch sizes etc.

## Conclusion

This documentation outlines the comprehensive steps taken to develop a legal text multilabel classification model. The implemented approaches, preprocessing steps, and model architecture aim to address specific challenges associated with legal texts.
The achieved results provide insights into the model's effectiveness in predicting violations of articles in the ECHR. The documentation serves as a guide for understanding the code, algorithms, and reasoning behind the decisions made during the development process.

