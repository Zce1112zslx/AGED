# 🎅AGED
Code and Technical Appendix for " Query Your Model with Definitions in FrameNet: An Effective Method for Frame Semantic Role Labeling" at AAAI-2023

## Data Download
Preprocessed FrameNet 1.5 and 1.7 data [link] (https://drive.google.com/drive/folders/1gk0Y1CW0V8JAuS3e0ihGFslJn0VATyYj)

## 🤖Technical Details

We use bert-base-uncased as the Pretrained Language
Model (PLM) in 🎅🏻AGED. We follow Chen, Zheng, and
Chang (2021); Zheng et al. (2022) to first train 🎅🏻AGED on exemplar sentences then train on the train set continually. We
search for hyperparameters (learning rate, batch size, and
epoch num) with performance in the development set. Performance
in the development set is also used to save the best
parameters of the models, and we will evaluate 🎅🏻AGED with
these parameters in the test set.

Our code is implemented with Pytorch and
Huggingface. 🎅AGED is trained on NVIDIA A40
with 40 GB memory and it will take about 4 GPU hours to
train 🎅🏻AGED and 0.6 hours when 🎅🏻AGED is trained only with
the training dataset. 

We show the hyper-parameter settings of 🎅🏻AGED in the following table.
| Hyper-parameter | Value|
| :--------------  | :----: |
| BERT version    | bert-base-uncased|
| batch size      | 16/<b>32</b>
| learning rate (train only) | 1e-5/<b>5e-5</b>
| learning rate (pretrain) | 5e-5 |
| learning rate (fine-tune) | <b>1e-5</b>/5e-5|
| lr scheduler    | Linear decay|
| warm up ratio   | 0.1|
| optimizer       | BertAdam|
| epoch num (train only) | 15/<b>20</b>/25|
| epoch num (pretrin) | <b>5</b>/10|
| epoch num (fine-tune) | <b>10</b>/20

Here, <b>"train only"</b> means 🎅🏻AGED trained only with training dataset. <b>"pretrain"</b> means 🎅🏻AGED w/exemplar trained with exemplar instances. <b>"fine-tune"</b> means 🎅🏻AGED w/exemplar then trained with training dataset.

We refer readers to [technical appendix](technical_appendix_8034.pdf) for more details.
