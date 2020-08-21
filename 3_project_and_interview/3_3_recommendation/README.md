# 推荐系统

## 基础

[推荐算法介绍](https://blog.csdn.net/u012050154/article/details/52267712)<br>

[recommend-learning github](https://github.com/littlemesie/recommend-learning)<br>
[book-recommendation-system](https://github.com/AbyssLink/book-recommendation-system)<br>

[深度学习推荐系统学习笔记](https://zhuanlan.zhihu.com/p/119248677?utm_source=zhihu&utm_medium=social&utm_oi=26827615633408)<br>


## 推荐相关比赛

这个比赛不错，适合推荐入门<br>
[Getting Started with a Movie Recommendation System](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/)<br>

这个教程也很不错<br>
[Recommender Systems in Python 101](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)<br>

 - Popularity model
 - Content-Based Filtering model
 - Collaborative Filtering model
   - Memory-based
     - user-based approach
     - item-based approach
   - Model-based
     - neural networks
     - bayesian networks
     - clustering models
     - latent factor models such as Singular Value Decomposition (SVD) and, probabilistic latent semantic analysis
 - Hybrid Recommender(cb+cf融合)

## 论文

**经典推荐论文**<br>
```text
FM：《Factorization Machines》
FFM：《Field-aware Factorization Machines for CTR Prediction》
DeepFM：《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》
Wide & Deep：《Wide & Deep Learning for Recommender Systems》
DCN：《Deep & Cross Network for Ad Click Predictions》
NFM：《Neural Factorization Machines for Sparse Predictive Analytics》
AFM：《Attentional Factorization Machines:Learning the Weight of Feature Interactions via Attention Networks》
GBDT + LR：《Practical Lessons from Predicting Clicks on Ads at Facebook》
MLR：《Learning Piece-wise Linear Modelsfrom Large Scale Data for Ad Click Prediction》
DIN：《Deep Interest Network for Click-Through Rate Prediction》
DIEN：《Deep Interest Evolution Network for Click-Through Rate Prediction》
BPR：《BPR: Bayesian Personalized Ranking from Implicit Feedback》
Youtube：《Deep Neural Networks for YouTube Recommendations》
```

**最近推荐论文，思维发散**<br>
```text
强化学习
《DRN: A Deep Reinforcement Learning Framework for News Recommendation》
《Deep Reinforcement Learning for List-wise Recommendations》

多任务学习
《Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate》
《Why I like it: Multi-task Learning for Recommendation and Explanation》

GAN
《IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models》
《CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks》

知识图谱

《DKN: Deep Knowledge-Aware Network for News Recommendation》
《RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems》
《Multi-task Learning for KG enhanced Recommendation》
《Perceive Your Users in Depth: Learning Universal User Representations from Multiple E-commerce Tasks》

Transformer
《Next Item Recommendation with Self-Attention》
《Deep Session Interest Network for Click-Through Rate Prediction》
《Behavior Sequence Transformer for E-commerce Recommendation in Alibaba》
《BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer》

RNN & GNN
《SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS》
《Improved Recurrent Neural Networks for Session-based Recommendations》
《Session-based Recommendation with Graph Neural Networks》

Embedding技巧
《Real-time Personalization using Embeddings for Search Ranking at Airbnb》
《Learning and Transferring IDs Representation in E-commerce》
《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》
```