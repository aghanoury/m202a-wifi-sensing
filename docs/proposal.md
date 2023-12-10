# Project Proposal

## 1. Motivation & Objective

WiFi sensing is a cost-effective and nonintrusive smart sensing solution that utilizes existing WiFi infrastructure. Unlike wearable sensors and camera-based solutions, it offers convenience and overcomes issues like occlusion, poor illumination, and privacy concerns. WiFi sensing has gained attention in ubiquitous computing research and supports applications like activity recognition, gesture recognition, and person identification, thanks to deep neural networks.

Because computational power at the edge may be limited, data is better off being transmitted to fog/cloud servers for processing. However, due to high sampling rates and dimensionality of CSI data acquire for HAR, the transmission of this information may seriously impact baseline WiFi performance. There is a need to explore sampling rates lower than the current state of the art. Lower sampling rates generates less data but will require developing new models for training and classification.

The goal of this project is to explore the use of various sampling rates for
human activity recognition (HAR) using Wifi channel state information (CSI).

## 2. State of the Art & Its Limitations

The initial project proposal provided some background research which explained that [1] suggests CSI sampling rate should be chosen as 800Hz for HAR as "typical human movement speed corresponds to CSI components of 300Hz", while [2] uses 500Hz and [3] uses 30Hz; [4] chooses 100Hz for indoor crowd counting; [5] exploits 200Hz for sign language recognition; etc.

Each one of these works makes different claims and assumptions about the rate of human activity movement which necessitates their use of a particular sampling rate. While the literature explores the use of various models, it fails to showcase concrete data on exploring various choices for sampling rate. Therefore, this work aims to explore both of those dimensions.

## 3. Novelty & Rationale

What is new in your approach and why do you think it will be successful?
Our approach will make use of the following datasets: UT_HAR, WIDAR, NTU-Fi_HAR, NTU-Fi_HumanID.

We will also be using the following models:MLP,LeNet, ResNet18, ResNet50, ResNet101, RNN, GRU, LSTM, BiLSTM, CNN+GRU, ViT.

Each of these datasets has a collection of various labeled and unlabeled samples of human activity at a particular sampling rate (find out which on each is using). One method of exploring other sampling rates would be to downsample a given set, then train our models on that particular set either as it is, or with various types of interpolation.

## 4. Potential Impact

If this project is successful, it may serve as the preliminary research needed to explore this topic even further and publish a novel and more efficient methodogoy for HAR via CSI.

## 5. Challenges

The main challenges of this work is not the changing of sampling rates itself, but training new models for classification. We will be constrained on time for this. For example, for every gesture we may want to explore _N_ sampling rates, thus for each of these we may want to then explore _M_ models. Each of these iterations requires significant time and resources for exploration. In general, there will be so many dimensions to our problem search that it might not be possible to explore enough solutions in the time we have.

## 6. Requirements for Success

To be successful in this project, we will need to have some familiarity (or gain it now!) in deep learning and the necessary frameworks for it's development. We currently have some of this experience.

## 7. Metrics of Success

We aim to achieve classification performance similar to that found in the literature.

## 8. Execution Plan

As it currently stands, we plan to first run trials using the recommended sampling rates and then with either downsampled or upsampled rates and then compare these results. Additionally, we will try different methods of downsampling and upsampling to see if results are different. We will run all the datasets against different models including MLP, LeNET, ResNet, etc.

If time permits we will explore what happens if the number of subcarriers change, if a different ML method is used, and using other datasets.

## 9. Related Work

### 9.a. Papers

See the [References](#references) section below.

### 9.b. Datasets

1. [NTU-Fi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) (HAR, Human ID) [6]
2. [UT-HAR](https://github.com/ermongroup/Wifi_Activity_Recognition), [7]
3. [Widar](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) (hand gesture recognition), [8]
4. [SignFi](https://github.com/yongsen/SignFi), [5]
5. [WiAR](https://github.com/linteresa/WiAR), [3]
6. [Exposing the CSI](https://github.com/ansresearch/exposing-the-csi), [9]
7. [RFDataFactory](https://www.rfdatafactory.com/datasets#wifi),
8. and other [datasets](https://github.com/Gi-z/CSI-Data).
   List datasets that you have identified and plan to use. Provide references
   (with full citation in the References section below).

### 9.c. Software

Python, PyTorch

## 10. References

List references corresponding to citations in your text above. For papers
please include full citation and URL. For datasets and software include name
and URL.

[1] Wang, Wei, et al. ["Understanding and modeling of wifi signal based human
activity recognition."](https://dl.acm.org/doi/abs/10.1145/2789168.2790093) MobiCom (2015).

[2] Yang, Jianfei, et al. ["EfficientFi: Toward large-scale lightweight WiFi
sensing via CSI compression."](https://ieeexplore.ieee.org/abstract/document/9667414) IEEE Internet of Things Journal (2022).

[3] Guo, Linlin, et al. ["Wiar: A public dataset for wifi-based activity
recognition."](https://ieeexplore.ieee.org/abstract/document/8866726) IEEE Access (2019).

[4] Hou, Huawei, et al. ["DASECount: Domain-Agnostic Sample-Efficient Wireless
Indoor Crowd Counting via Few-Shot Learning."](https://ieeexplore.ieee.org/abstract/document/9996126) IEEE Internet of Things Journal
(2022).

[5] Ma, Yongsen, et al. ["SignFi: Sign language recognition using WiFi."](https://dl.acm.org/doi/abs/10.1145/3191755) ACM
IMWUT (2018).

[6] Yang, Chen, et al. ["SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing"](https://arxiv.org/abs/2207.07859), Patterns (2023).

[7] Yousefi, Narui, et al. ["A Survey on Behavior Recognition Using WiFi Channel State Information"](https://ieeexplore.ieee.org/document/8067693)IEEE Comunication Magazine (2017)

[8] Zheng Yang, Yi Zhang, Guidong Zhang, Yue Zheng, December 26, 2020, "Widar 3.0: WiFi-based Activity Recognition Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/7znf-qp86.

[9] M. Cominelli, F. Gringoli and F. Restuccia, "Exposing the CSI: A Systematic Investigation of CSI-based Wi-Fi Sensing Capabilities and Limitations," 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom), Atlanta, GA, USA, 2023, pp. 81-90, doi: 10.1109/PERCOM56429.2023.10099368.
