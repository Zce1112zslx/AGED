# 👴🏻AGED
Code and Technical Appendix for " Query Your Model with Definitions in FrameNet: An Effective Method for Frame Semantic Role Labeling" at AAAI-2023


## 🤖Technical Details
We show the hyper-parameter settings of 👴🏻AGED in the following table.
| Hyper-parameter | Value|
| --------------  | :----: |
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

Here, <b>"train only"</b> means 👴🏻AGED trained only with training dataset. <b>"pretrain"</b> means 👴🏻AGED w/ exemplar trained with exemplar instances. <b>"fine-tune"</b> means 👴🏻AGED w/exemplar then trained with training dataset.

We refer readers to [technical appendix](technical_appendix_8034.pdf) for more details.