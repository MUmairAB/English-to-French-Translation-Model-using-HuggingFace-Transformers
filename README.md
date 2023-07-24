# English to French Translation Model using HuggingFace Transformers

Sequence-to-sequence tasks have gained significant attention in NLP due to their ability to handle a variety of language processing challenges. Translation, as one such sequence-to-sequence task, plays a crucial role in bridging language barriers and facilitating cross-cultural communication. Additionally, sequence-to-sequence models can be applied to other tasks like style transfer and generative question answering.

In this project, we will concentrate on training a powerful language translation model using HuggingFace Transformers. The specific objective is to create a model that can accurately translate English sentences to their corresponding French translations.

## 1. Dataset

For the fine-tuning of our translation model, we will be using the [KDE4 dataset](https://huggingface.co/datasets/kde4) available from HuggingFace. The KDE4 dataset contains a large parallel corpus of English and French sentences, making it suitable for training a robust translation model.

## 2. Methodology

The methodology involves the following key steps:

#### 2.1 Data Preprocessing:
We will preprocess the KDE4 dataset to handle any inconsistencies, tokenize the sentences, and prepare them for training.

#### 2.2 Fine-tuning:
For this task, we will use the HuggingFace Transformers library, which offers a range of pre-trained transformer models. The model that we'll fine-tune is [Helsinki-NLP's English to French Translator](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr?text=My+name+is+Sarah+and+I+live+in+London).

#### 2.3 Model Evaluation:
To ensure the quality of translations, we will evaluate the fine-tuned English to French translation model using SacreBLEU, a popular and modified version of the BLEU score specifically designed for text translation tasks.

#### 2.4 Inference:
The fine-tuned English to French translation model is now deployed on the [HuggingFace Model Hub](https://huggingface.co/MUmairAB/marian-finetuned-kde4-english-to-french), making it easily accessible for anyone to use. Additionally, a [HuggingFace Space](https://huggingface.co/spaces/MUmairAB/English-to-French) has been created for this model, simplifying the process of utilizing the translation capabilities.


## 3. Training hyperparameters

The following hyperparameters were used during training:
```
optimizer: {'name': 'AdamWeightDecay', 'learning_rate': {'class_name': 'PolynomialDecay', 'config': {'initial_learning_rate': 5e-05, 'decay_steps': 29555, 'end_learning_rate': 0.0, 'power': 1.0, 'cycle': False, 'name': None}}, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'amsgrad': False, 'weight_decay_rate': 0.01}

training_precision: float32
```

## 4. Usage

To utilize the English to French translation model, users can follow the steps below:

```
# Use a pipeline as a high-level helper
from transformers import pipeline

#Define the model checkpoint
model_checkpoint = "MUmairAB/marian-finetuned-kde4-english-to-french"

#Instantiate the model
translator = pipeline("translation", model=model_checkpoint)

#English text to be translated
en_text = "French is a very difficult language."

#Apply the model
translator(en_text)
```
