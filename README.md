# &nbsp; ![Reinforce-Joey](reinforce_joey.png) Reinforce-Joey
This is a fork of the awesome Joey NMT with implementations of several Reinforcement Learning algorithms, reward functions and baselines.  
This is also the code for [Revisiting the Weaknesses of Reinforcement Learning for Neural Machine Translation](https://arxiv.org/abs/2106.08942).   
If you have any further questions about the paper or on how to use the code, feel free to send an email. 

## Implemented algorithms:  
The forward pass of each method can be found here:  
[REINFORCE](https://github.com/samuki/reinforce-joey/blob/3b12dfe40687155d95d7f45608be90595866d542/joeynmt/model.py#L80), [MRT](https://github.com/samuki/reinforce-joey/blob/3b12dfe40687155d95d7f45608be90595866d542/joeynmt/model.py#L153), [NED_A2C](https://github.com/samuki/reinforce-joey/blob/3b12dfe40687155d95d7f45608be90595866d542/joeynmt/model.py#L250)   
 
 The implemented papers can be found here:  
 
- Policy Gradient aka REINFORCE as in [Kreutzer et al. (2017)](https://www.aclweb.org/anthology/P17-1138/)
- Minimum Risk Training as in [Shen et al. (2016)](https://www.aclweb.org/anthology/P16-1159/)
- Advantage Actor-Critic aka NED-A2C as in [Nguyen et al. (2017)](https://www.aclweb.org/anthology/D17-1153/)

Policy Gradient and MRT can be used with both Transformers and RNNs but NED-A2C is currently only implemented for RNNs. 

## How to use 
In general cold-starting a model with Reinforcement Learning is difficult as the methods rely on random sampling.  
That means to effectively use the algorithms you have to pretrain a Transformer/RNN  and then fine-tune it with RL.  
Pretrained models for Joey NMT are available [here](https://github.com/joeynmt/joeynmt/blob/master/README.md#pre-trained-models) and the models used in the paper can be downloaded [here](https://drive.google.com/drive/folders/1lptEFQeo6wIsNan-7MtKwk3WYRIyOAJd?usp=sharing). 

## Parameters
The method and corresponding hyperparameters are specified in the config, examples for PG and MRT can be found in the [configs directory](https://github.com/samuki/reinforce-joey/blob/master/configs), e.g. [pg_cross_domain.yaml](https://github.com/samuki/reinforce-joey/blob/master/configs/pg_cross_domain.yaml), [mrt_cross_domain.yaml](https://github.com/samuki/reinforce-joey/blob/master/configs/mrt_cross_domain.yaml).  
Here is a short explanation of the parameters:
* All methods: 
  * temperature: Softmax temperature parameter. Decreasing results in a 'peakier' distribution and more exploitation while increasing leads to more exploration.  
* Policy Gradient/Reinforce:   
  * reward: 
    * bleu: standard corpus_bleu from sacrebleu
    * scaled_bleu: scale the BLEU locally to the interval [-0.5, 0.5] for each batch
    * constant: constant reward of 1 
  * baseline: 
    * False: no baseline
    * average_reward_baseline: subtract a running average of all previous BLEUs from the rewards
    * learned_reward_baseline: train a regression network to estimate BLEUs (usually worse than average bl., currently only on the learned_reward branch)
* MRT:  
  * add_gold: add gold/reference to sample space
  * samples: number of samples
  * alpha: mrt smoothness parameter 
* Advantage Actor-Critic:  
  * critic_learning_rate: learning rate of critic network

## Environment 
The experiments were conducted on Tesla V100 GPUs using Python 3.8.5, PyTorch 1.7.0 and CUDA 11.  
For a detailed explanation on how to install Joey NMT, please refer following [instructions](https://github.com/joeynmt/joeynmt#installation)


## Acknowledgements
Thanks a lot to [Michael Staniek](https://github.com/MStaniek) who was a great help with implementation details and his knowledge about Reinforcement Learning.

## Reference
Our paper at NAACL 2021: 
```
@inproceedings{kiegeland-kreutzer-2021-revisiting,
    title = "Revisiting the Weaknesses of Reinforcement Learning for Neural Machine Translation",
    author = "Kiegeland, Samuel  and
      Kreutzer, Julia",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.133",
    pages = "1673--1681",
    abstract = "Policy gradient algorithms have found wide adoption in NLP, but have recently become subject to criticism, doubting their suitability for NMT. Choshen et al. (2020) identify multiple weaknesses and suspect that their success is determined by the shape of output distributions rather than the reward. In this paper, we revisit these claims and study them under a wider range of configurations. Our experiments on in-domain and cross-domain adaptation reveal the importance of exploration and reward scaling, and provide empirical counter-evidence to these claims.",
}
```
Joey NMT: 
```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```
