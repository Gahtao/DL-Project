# Deep Learning Group Assignment
## Introduction
This repository is structured as followed:
1. The `.gitignore` file blocks the `data` folder and anything in it to be uploaded to GitHub. This is to prevent bloat from large files. Download the files manually from convas and extract them there. The folder structure should looks something like this:
   - data\
     - Test set\
       - 1035.wav
       - 1074.wav
       - ...
     - Train\
       - 1f_1018.wav
       - 1f_1026.wav
       - ...
2. Two folders are pre-made, `model A` and `model B`. The model A folder is for Anna and Jess to do all their work in, and the `model B` is for Bara, Gaia, and Stefan, as per the task division. Further information about the exact assignment can be found below or on Canvas.
3. Firstly, feel free ot ask, but here is a refresher on GitHub:
   1. `git clone https://github.com/Gahtao/DL-Project.git` to copy this repository to your PC.
      - Don't forget to move into the newly created folder before continuing.
   2. `git pull` to update your local files. Use this command _always_ **before** you start working on the project.
   3. `git push` to update the online files. Use this command _always_ **after** you are done working on the project.
   4. Sometimes these commands may throw a fit because multiple people have worked on the same file, at the same time, or someone didn't use the above commands structurally as described. In that case, send a screenshot of whatever message pops up in the group chat (or resolve it yourself if you have the know-how).

## Set-Up
I'll probably automate a conda environment for reproducibility, but not right now.

## 1. Shared Task – Accent classification of Spoken Language using
The data set you will work with for this assignment is a collection of utterances recorded in a controlled environment. This data set is rich and diverse, featuring utterances from multiple male and female speakers from different countries. The sentences are in the same language but with five different accents. Each utterance is stored in a .wav file and is annotated with the corresponding accent and gender of the speaker.
### 1.1 Data format
Here are the key characteristics of the dataset:
- **Utterances**: The dataset comprises various utterances from different actors. They are recorded with a sampling rate of 16000 and are single channel.
- **Annotations**: Each audio file is annotated with the corresponding accent and the speaker’s gender. The accent is encoded in the first character of the file name with a single number from 1 to 5. The gender is encoded as a single letter (’m’ or ’f’) corresponding to the second character of the file name.

Each recording varies in length, so your model must be capable of handling inputs of different lengths. The data set will be provided in a ZIP file containing a folder with all recordings in .wav format. As explained above, annotations are embedded in the file names.

Your final model must be evaluated on a given test set that will be provided later. The test set only contains the recordings, so no annotations are given. You have to submit your classification results to a competition website to get your model’s performance on that set.

Your goal is to build a Deep Learningmodel that is capable of estimating the accent of a recording. Understanding your dataset is the first step towards creating a successful model. So, take your time to explore and understand the data.

### 1.2 Required experiments
To achieve the goals of this assignment, you must ensure you complete the following:
1. **Model Comparisons**:
   - Train and compare the performance of (at least) these two approaches:
     1. Use the raw input signal and analyze it as a 1D input; in this case, you can, of course, apply simple pre-processing like standardization.
     2. Use a transformation or feature engineering and use the transformed version as the input to the model. In this case, the format of the signal can be different from the original one (e.g., Short Time Fourier Transform).
   - For each approach, you can use different deep learning architectures (e.g., Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, etc.). 
   - Analyze and discuss the strengths and weaknesses of each approach and architecture.
2. **Regularization Techniques**: Implement and evaluate the impact of regularization techniques such as dropout, batch normalization, or weight decay. Explain how these techniques influence model performance and generalization.
3. **Performance Evaluation**: Assess the classification performance of your models using appropriate metrics such as accuracy, precision, recall, and F1-score. Additionally, analyze performance regarding the following:
   - By class (accent): Determine if the model performs better on certain accents than others and explore potential reasons. 
   - By speaker’s gender: Investigate whether gender biases exist in model predictions.
4. **Data Augmentation**: Implement data augmentation techniques that match your approach. Evaluate how these techniques impact model performance and whether they improve generalization.