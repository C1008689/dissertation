# Sentiment Analysis in Literature

## Dissertation Project

**Sheffield Hallam University**

**Department of Engineering and Mathematics**

**BSc (Hons) Mathematics**

**Project Title:** Sentiment Analysis in Literature

**Student:** Aqib Ullah

**Supervisor:** Dr. Keith Harris

**Moderator:** Ros Porter

---

## Abstract

This repository contains the code and resources used for my dissertation on sentiment analysis of Amazon book reviews. The study employs a blend of traditional machine learning and advanced artificial intelligence techniques to understand consumer sentiments. Key methodologies include Term Frequency-Inverse Document Frequency (TF-IDF), Word Embeddings, Naive Bayes, and Stochastic Gradient Descent Classifier (SGDClassifier). The research highlights the enhanced performance of deep learning models in capturing nuanced emotional expressions and integrates semantic analysis to improve context interpretation.

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Ethical Considerations](#ethical-considerations)
7. [Acknowledgements](#acknowledgements)
8. [References](#references)

## Introduction

The digital age, led by platforms like Amazon, has significantly altered consumer behavior, particularly in how experiences and opinions are shared online. Sentiment analysis, a branch of natural language processing (NLP), enables the automated categorization of opinions within text, facilitating a deeper understanding of customer sentiments and preferences. This dissertation focuses on sentiment analysis of Amazon book reviews, evaluating various sentiment analysis methods from traditional machine learning to sophisticated deep learning techniques.

## Literature Review

The dissertation builds on foundational works in sentiment analysis, including:
- Pang and Lee's "Opinion Mining and Sentiment Analysis"
- Hutto and Gilbert's "vader: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text"
- Blei et al.'s "Latent Dirichlet Allocation"
- Du Sautoy's "The Creativity Code"
- Larson's "The Myth of Artificial Intelligence"

These sources provide a deep background in sentiment analysis methodologies, the evolution of machine learning models, and the ethical implications of AI.

## Methodology

### Data Collection and Preprocessing

Data was collected from Amazon book reviews in JSON format. Key preprocessing steps included:
- JSON Parsing
- Text Cleaning
- Tokenization and Lemmatization
- Stop Word Removal
- Vectorization

### Model Training

Models trained include Naive Bayes, SGDClassifier, and a Long Short-Term Memory (LSTM) network. The LSTM network, in particular, was trained on preprocessed text data, leveraging word embeddings and deep learning techniques to capture contextual dependencies.

### Evaluation

Model performance was evaluated using accuracy, precision, recall, F1-score, and ROC curve analysis. The LSTM model demonstrated significant improvements in capturing nuanced sentiments compared to traditional methods.

## Results

The analysis revealed a high level of satisfaction among readers, with 79% positive reviews and an average rating score of 4.22. Model performance metrics indicated that the LSTM network achieved an accuracy of approximately 83%, highlighting its effectiveness in sentiment analysis tasks.

## Conclusion

This research contributes to the field of sentiment analysis by presenting a robust framework for analyzing consumer sentiments in online book reviews. The findings offer valuable insights for authors, publishers, and e-commerce platforms, informing strategies for content creation, marketing, and customer engagement.

## Ethical Considerations

User data anonymity and privacy were prioritized throughout the study, in accordance with the Data Protection Act 2018 and GDPR guidelines. Ethical considerations are detailed in Appendix E of the dissertation.

## Acknowledgements

I would like to express my deepest gratitude to Dr. Keith Harris for his unwavering support and guidance throughout this dissertation. I also extend my thanks to my family, friends, and peers for their encouragement and understanding.

## References

A comprehensive list of references is provided in the dissertation document, covering all sources and literature reviewed in the study.
