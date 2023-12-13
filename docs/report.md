## Table of Contents

- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Background and Related Work](#2-background-and-related-work)
  - [Channel State Information (CSI)](#channel-state-information-csi)
  - [Feature Extraction](#feature-extraction)
    - [STFT](#stft)
    - [BVP](#bvp)
    - [General Feature Extractions](#general-feature-extractions)
  - [Deep Learning Models](#deep-learning-models)
  - [Motivation](#motivation)
- [3. Technical Approach](#3-technical-approach)
- [4. Evaluation and Results](#4-evaluation-and-results)
- [5. Discussion and Conclusions](#5-discussion-and-conclusions)
- [6. References](#6-references)

## Abstract

Recent studies have developed different sensing applications like human activity
recognition (HAR) using WiFi channel state information (CSI) information.
However, they usually use high and different sampling rates of CSI, which is
impractical and will hurt the communication performance. Besides, different
sensing tasks or applications may require different minimum/best sampling rates
due to different movement speeds and highest frequency of the activities. E.g,
[1] suggests CSI sampling rate should be chosen as 800Hz for HAR, while [2] uses
500Hz and [3] uses 30Hz; [4] chooses 100Hz for indoor crowd counting; [5]
exploits 200Hz for sign language recognition; etc. Therefore, it’s interesting
to explore different applications’ dependence on sampling rates. In this study,
we perform a comprehensive analysis of these datasets, changing sampling rate
and observing the impact on the accuracy of the model. Our results find that each
dataset has ample room to reduce sampling rate without sacrificing accuracy.

## 1. Introduction

This section should cover the following items:

- Motivation & Objective: What are you trying to do and why? (plain English without jargon)
- State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
- Novelty & Rationale: What is new in your approach and why do you think it will be successful?
- Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
- Challenges: What are the challenges and risks?
- Requirements for Success: What skills and resources are necessary to perform the project?
- Metrics of Success: What are metrics by which you would check for success?

Human Activity Recognition (HAR) is the process of identifying and interpreting
the actions and activities performed by humans through the analysis of data
collected from various sensors. This includes vision-based methods, Inertial
Measurement Units (IMU), microphones, and the focus of this work, Radio
Frequency (RF) sensors. In particular, WiFi-based HAR leverages native Multiple
Input Multiple Output (MIMO) Channel State Information (CSI) to detect changes
in WiFi signal strength. The goal of this research is to concentrate on
WiFi-based methods using pre-existing datasets. The notable contribution of this
work lies in the discovery that each dataset examined provides significant room
for sample reduction without compromising performance. This finding has
implications for optimizing data collection and processing in HAR, enhancing its
efficiency and practical applicability.

![HAR](media/general_gestures.png)
*Figure 1: An overview of various gestures recognizable from [??] dataset.*

The initial project proposal provided some background research which explained
that [1] suggests CSI sampling rate should be chosen as 800Hz for HAR as
"typical human movement speed corresponds to CSI components of 300Hz", while [2]
uses 500Hz and [3] uses 30Hz; [4] chooses 100Hz for indoor crowd counting; [5]
exploits 200Hz for sign language recognition; etc.

Each one of these works makes different claims and assumptions about the rate of
human activity movement which necessitates their use of a particular sampling
rate. While the literature explores the use of various models, it fails to
showcase concrete data on exploring various choices for sampling rate.
Therefore, this work aims to explore both of those dimensions.

## 2. Background and Related Work


### Channel State Information (CSI)
The choice of WiFi-based Human Activity Recognition (HAR) is motivated by
distinct advantages over traditional camera-based systems. Camera systems, while
accurate and deterministic, pose limitations due to their requirement for direct
line-of-sight (LOS), raising privacy concerns and potential intentional evasion.
In contrast, WiFi, being ubiquitous in indoor settings, offers a compelling
alternative for HAR. WiFi-based HAR operates passively, mitigating privacy
concerns associated with cameras and eliminating the need for a direct line of
sight. This not only addresses privacy issues but also provides a more inclusive
and less intrusive method for detecting and interpreting human activities in
indoor environments. The widespread presence of WiFi further enhances the
practicality and accessibility of this approach, making it a valuable option for
activity recognition applications.

![Priv](media/privacy_centric.png)
*Figure ?: On the top: a traditional vision-based tracking system. On the bottom: WiFi based HAR system.*

Channel State Information (CSI) stands as a superior alternative to Received
Signal Strength (RSS) for Human Activity Recognition (HAR) due to its capacity
to provide a more nuanced and comprehensive depiction of the WiFi environment.
Unlike RSS, which merely averages signal strength across the entire bandwidth,
CSI, obtained through tools like Intel NIC or Atheros CSI, encapsulates the
amplitude and phase of each channel. This yields a richer dataset for HAR,
essentially creating a detailed "WiFi Image" that captures the intricacies of
signal propagation. While RSS is limited by its oversimplified representation,
CSI, by measuring the Channel Impulse Response (CIR) in the frequency domain,
accounts for factors such as amplitude, phase, time delay, and multipath
components. This shift toward CSI as the preferred metric reflects its ability
to offer a more detailed and accurate insight into the WiFi landscape, enhancing
the efficacy of HAR applications in diverse and challenging environments.

### Feature Extraction
Feature extraction is a process in which relevant information or features are
selected or extracted from raw data to reduce its dimensionality or to transform
it into a more suitable format for analysis. It's important because it reduces
the dimensionality of data, allowing machine learning algorithms to focus on the
most relevant information and improving computational efficiency.  By capturing
essential patterns or characteristics, it enhances the performance of models in
various domains, such as image recognition, natural language processing, and
signal processing.

#### STFT
In the context of WiFi-based HAR, directly feeding CSI data into a model proves
impractical. The necessity for effective feature extraction is apparent, and
various approaches are employed to enhance the interpretability of the data. One
common method, and that which is employed in the UT-HAR dataset [??], involves
the application of Short-time Fourier transforms (STFT).  This technique proves
invaluable in dissecting the distinct phases of movements embedded within the
CSI data. By extracting relevant features through STFT, the model gains a more
refined understanding of the temporal dynamics inherent in human activities.
This preliminary step in feature extraction serves as an important preprocessing
stage, facilitating the subsequent stages of model training and improving the
overall accuracy and effectiveness of HAR systems.

![STFT](media/stft_example.png)
*Figure 2: Another signal*

#### BVP
The Widar dataset utilizes the Body-coordinate velocity profile (BVP) as a key
component in its analysis. The data processing pipeline involves two major
stages following the acquisition of Channel State Information (CSI): first,
converting CSI to BVP, and then extracting relevant features from the BVP. The
subsequent work involves classifying activities based on this Body-coordinate
velocity profile, showcasing the significance of feature extraction in
discerning patterns and activities from wireless signal data.

![BVP](media/bvp_pipeline.png)
*Figure 3: BVP pipeline from [??]*

#### General Feature Extractions
EfficientFi, authored by the creators of SenseFi and NTU datasets, employs a
pipeline involving feature extraction directly from raw CSI data. They utilize a
quantization method to compress the feature map by mapping measured feature
vectors to the nearest vector in a CSI codebook. Classification is performed
solely on the extracted features, and a decoder network is employed to store CSI
data on the server itself, emphasizing the classification based on
feature-extracted data.

![NTU](media/NTU_feature_extraction.png)
*Figure ??: architecture of NTU-Fi*

### Deep Learning Models
**MLP** (Multi-Layer Perceptron): Simple and robust architecture, but slow
*convergence and significant computational costs are drawbacks.

**CNN** (Convolutional Neural Network): Excels in capturing spatial and temporal
*features, but may have a limited receptive field due to kernel size and
*traditionally stacks all feature maps equally.

**RNN** (Recurrent Neural Network): Effective for handling time sequence data
*like video and Channel State Information (CSI), capable of memorizing
*arbitrary-length sequences. However, faces challenges in capturing long-term
*dependencies and suffers from the vanishing gradient problem during
*backpropagation.

**LSTM** (Long Short-Term Memory): Addresses the vanishing gradient problem in
*traditional RNNs, allowing better handling of long-term dependencies. However,
*it introduces increased complexity compared to standard RNNs.

### Motivation
The motivation involves gathering Channel State Information (CSI) at the edge, followed by post-processing, potentially including denoising, and feature extraction using methods like Short-Time Fourier Transform (STFT) and Velocity Profile. The feature data is then offloaded to servers for machine learning classification. The primary goal is to analyze the impact of sampling rate on classification performance, with the potential benefit of minimizing traffic between the edge and the cloud.

![motivation](media/motivation.png)
*Figure ??: General WiFi HAR processing architecture with emphasis where our work explores*

## 3. Technical Approach

## 4. Evaluation and Results

## 5. Discussion and Conclusions

## 6. References
