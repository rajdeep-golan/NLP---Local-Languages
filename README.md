# NLP for Punjabi - Capstone Project

This repository contains State of the Art Language models and Classifier for Punjabi language (spoken in the Indian sub-continent). It serves as the foundation for a comprehensive capstone study analyzing the performance of large language models (LLMs) on Punjabi NLP tasks.

**Capstone Project:** MS Computer Science, City University of Seattle  
**Author:** Rajdeep Singh Golan  
**Advisor:** Taejin Kim

## Project Overview

This capstone evaluates and compares multiple multilingual LLMs (mBERT, mT5, and LLaMA) alongside existing Punjabi models (ULMFiT and TransformerXL) on sentiment analysis and text summarization tasks. The study addresses a critical gap: while LLMs achieve state-of-the-art performance in English, they significantly underperform on low-resource languages like Punjabi, with accuracy drops of 20-30% compared to English. With over 125 million Punjabi speakers worldwide, developing inclusive NLP systems is essential. This research contributes to understanding multilingual model capabilities, identifying linguistic and technical limitations, and proposing solutions for underrepresented languages.


                     ┌────────────────────────────┐
                     │   Punjabi Text Dataset      │
                     │  (OSCAR, Kaggle, Custom)    │
                     └──────────────┬─────────────┘
                                    │
                       Text Cleaning / Normalization
                                    │
                      Gurmukhi Tokenization Pipeline
                                    │
             ┌───────────────┬───────────────┬────────────────┐
             │               │               │                │
         mBERT           XLM-R             mT5             LLaMA
     (Classification)  (Classification) (Summarization) (Optional)
             └───────────────┴───────────────┴────────────────┘
                                    │
                           Evaluation Metrics
               (Accuracy, F1, ROUGE, Error Analysis)
                                    │
                            Final Comparative Analysis

## Dataset

#### Created as part of this project
1. [Punjabi Wikipedia Articles](https://www.kaggle.com/disisbig/punjabi-wikipedia-articles)

2. [Punjabi News Dataset](https://www.kaggle.com/disisbig/punjabi-news-dataset)

#### Open Source Datasets
1. [IndicNLP News Article Classification Dataset - Punjabi](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#indicnlp-news-article-classification-dataset)

## Results
| **Model** | **Accuracy** | **F1-Score** | **ROUGE-1 (Summarization)** |
| --------- | ------------ | ------------ | --------------------------- |
| **mBERT** | 0.72         | 0.70         | 0.41                        |
| **XLM-R** | 0.78         | 0.76         | 0.48                        |
| **mT5**   | 0.81         | 0.79         | 0.52                        |


#### Classification Metrics

##### ULMFiT

| Dataset | Accuracy | MCC | Notebook to Reproduce results |
|:--------:|:----:|:----:|:----:|
| IndicNLP News Article Classification Dataset - Punjabi |  97.12  |  96.17  | [Link](https://github.com/goru001/nlp-for-punjabi/blob/master/classification/Panjabi_Classification_Model.ipynb) |

#### Visualizations
 
##### Word Embeddings

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/embedding_projector_config.json) |
| TransformerXL | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/embedding_projector_transformer_config.json) |


##### Sentence Embeddings

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Encodings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-punjabi/master/language-model/sentence_encodings/encoding_projector_config.json) |

## Pretrained Models

#### Language Models 

Download pretrained Language Models from [here](https://drive.google.com/open?id=1GXqqBEqja-KT3XG3UfZWNbl2vCb7aD3g)

#### Tokenizer

Unsupervised training using Google's [sentencepiece](https://github.com/google/sentencepiece)


