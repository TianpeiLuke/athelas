# Temporal Self-Attention Model
Temporal Self-Attention is a variant of self-attention that makes the model pay attention to orders based on both time and order information. We use it as a building block to automatically generate features with different entities` (e.g., payment token, IP) sequence. We also include one self-attention module to learn the interaction of features aggregated from a sequence (denoted as “Feature-Attention”), which acts as a bridge for engineered features before we can fully automate feature engineering.  Overall, we refer to the full model as the TSA model.
![TSA](./figs/orderfeatureattention.png)

---
## Invoking SageMaker Training Job
There are two options to invoke the SageMaker training job. 
- Run notebook `MODS/TSA_pytorch_processor_train.ipynb`, 
which is better suitable for experiments. It is also easy to directly run the training experiment with local notebook instance by running `MODS/tsa/scripts/train.py` with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). 
- Launch SageMaker pipeline, e.g., config the pipeline with `MODS/tsa/eu_tsa_sq_model.py` 
and invoke the pipeline with `MODS/tsa/eu_tsa_model_mods_launch.ipynb`, which is for production deployment.

### SageMaker Pipeline
Taking EU TSA Model as example. Invoking pipeline by running `eu_tsa_model_mods_launch.ipynb`. Install mods and sais workflow packages with first cell (uncomment first)
if notebook instance just got started. 

Main components of the pipeline:
- Date downloading:
  - Filters set in `eu_tsa_model_mods_launch.ipynb`
- Preprocessing:
  - Processing files by chunks with function `chunk_processing`, with multi-processing for each chuck
  - Missing values will be filled with the feature specific map `config/default_value_dict.json`
  - Sequence categorical features will be encoded to integers (row index in embedding table) using map `config/cat_to_index.json`
  - Sequence and engineered numerical features will be min-max scaled using `config/preprocessor.pkl`
- Training:
  - Distributed Data Parallel with multi instance, multi GPU
  - Automatic Mixed Precision Training
  - Mixture-of-experts (MoE) option for putting sparse MoE inside transformer layer
  - Percentile mapping generated
- generic_rfuge: 
  - Probability score generated
- AddInferenceDependencies:
  - Sagemaker model packing
- MimsModelRegistrationProcessingStep:
  - Model registration and auto load testing
  - Can manually set auto scaling policy with last cell in `eu_tsa_model_mods_launch.ipynb`
![MODS](./figs/eu_tsa_sq_pipeline.png)
---
## Model Structure in detail
The model structure is highly modularized, main components are listed here:

- [OrderFeature Model](#orderfeature-model)
  - [OrderFeatureAttentionClassifier](#orderfeatureattentionclassifier)
  - [TwoSeqMoEOrderFeatureAttentionClassifier](#twoseqmoeorderfeatureattentionclassifier)
    - [OrderAttentionLayer](#orderattentionlayer)
      - [AttentionLayer](#attentionlayer)
      - [FeatureAggregation](#featureaggregation)
    - [FeatureAttentionLayer](#featureattentionlayer)
      - [AttentionLayerPreNorm](#attentionlayerprenorm)

### OrderFeature Model 
- TSA Model (`OrderFeatureAttentionClassifier`) with both OrderAttention and FeatureAttention module operating on single sequence data
- Datasets
    - `train`: used for training
    - `vali`: used for calculating AUC and early stopping
    - `cali`: used for percentile score mapping (only for production)
- Input for each dataset:
    - `X_seq_cat`: Categorical sequence data, dimension (``n_data_points``, ``seq_len``, ``n_cat_features``), where ``n_data_points`` is number of samples, ``seq_len`` is sequence length and ``n_cat_features`` is number of categorical features. The categorical values have been encoded to row index in a embedding lookup table
    - `X_seq_num`: Numerical sequence data, dimension (``n_data_points``, ``seq_len``, ``n_num_features``+2), where ``n_num_features`` is number of numerical features, and the extra two dimensions are corresponding to epoch order date and toKeep flag (1-isPadding).
    - `X_num`: Numerical dense engineered data, dimension (``n_data_points``, ``dim_embedding_table``), where ``dim_embedding_table`` is number of numerical engineered features.
    - `Y`: binary label vector with length ``n_data_points``

### Two Sequence MoE OrderFeature Model
- Two Sequence TSA Model (`TwoSeqMoEOrderFeatureAttentionClassifier`) with OrderAttention module operating on customerId (cid) and 
creditCardIds (ccid) sequences, combined with a gate function, and FeatureAttention on current order's sequence and 
engineered features.
- Datasets
    - `train`: used for training
    - `vali`: used for calculating AUC and early stopping
    - `cali`: used for percentile score mapping (only for production)
- Input for each dataset:
    - `X_seq_cat_cid`: Categorical sequence data for cid sequence, dimension (``n_data_points``, ``seq_len``, ``n_cat_features``), where ``n_data_points`` is number of samples, ``seq_len`` is sequence length and ``n_cat_features`` is number of categorical features. The categorical values have been encoded to row index in a embedding lookup table
    - `X_seq_num_cid`: Numerical sequence data for cid sequence, dimension (``n_data_points``, ``seq_len``, ``n_num_features``+2), where ``n_num_features`` is number of numerical features, and the extra two dimensions are corresponding to epoch order date and toKeep flag (1-isPadding).
    - `X_seq_cat_ccid`: Categorical sequence data for ccid sequence, dimension (``n_data_points``, ``seq_len``, ``n_cat_features``), where ``n_data_points`` is number of samples, ``seq_len`` is sequence length and ``n_cat_features`` is number of categorical features. The categorical values have been encoded to row index in a embedding lookup table
    - `X_seq_num_ccid`: Numerical sequence data for ccid sequence, dimension (``n_data_points``, ``seq_len``, ``n_num_features``+2), where ``n_num_features`` is number of numerical features, and the extra two dimensions are corresponding to epoch order date and toKeep flag (1-isPadding).
    - `X_num`: Numerical dense engineered data, dimension (``n_data_points``, ``dim_embedding_table``), where ``dim_embedding_table`` is number of numerical engineered features.
    - `Y`: binary label vector with length ``n_data_points``
![TwoSeqTSA](./figs/na_tsa_sq_model.png)

### ``MODS/tsa/scripts/models.py``

#### OrderFeatureAttentionClassifier
- Inputs:
  - `x_cat`: Categorical sequence features of shape (`batch_size`, `seq_len`, `n_cat_features`)
  - `x_num`: Numerical sequence features of shape (`batch_size`, `seq_len`, `n_num_features`)
  - `x_engineered`: Numerical engineered features of shape (`batch_size`, `dim_embedding_table`)
  - `time_seq`: Time delta sequence
  - `attn_mask`: Attention mask (optional)
  - `key_padding_mask`: Key padding mask (optional)
- Outputs:
  - `scores`: Output class scores, shape (`batch_size`, `n_classes`)
  - `ensemble`: Ensemble embeddings, shape (`batch_size`, `3 * dim_embedding_table` (+ `dim_embedding_table` if use_mlp))
- Parameters:
  - `n_cat_features`: Number of categorical sequence features 
  - `n_num_features`: Number of numerical sequence features 
  - `n_classes`: Number of output classes
  - `n_embedding`: Size of sequence embedding table
  - `seq_len`: int - Sequence length
  - `n_engineered_num_features`: int - Number of numerical engineered features 
  - `dim_embedding_table`: int - Dimension of embedding table
  - `dim_attn_feedforward`: Dimension of feedforward network inside transformer layer
  - `use_mlp`: Whether to use MLP on numerical features to produce part of the embeddings
  - `num_heads`: Number of attention heads
  - `dropout`: Dropout probability (default 0.1)
  - `n_layers_order`: Number of order attention layers
  - `n_layers_feature`: Number of feature attention layers
  - `emb_tbl_use_bias`: Whether to use bias term to embeddings
  - `use_moe`: Whether to use mixture-of-experts (MoE) structure for transformer layer
  - `num_experts`: Number of experts to use for MoE layer
  - `use_time_seq`: Whether to use time information
  - `return_seq`: Whether to return sequence of embeddings
- Forward Pass
  - Apply order attention
  - Apply feature attention 
  - Optionally apply MLP on numerical features
  - Concatenate order, feature, and MLP embeddings
  - Feed concatenated embeddings to classifier to produce scores

#### TwoSeqMoEOrderFeatureAttentionClassifier
- Inputs:
  - `x_seq_cat_cid`: Categorical sequence features for cid sequence of shape (`batch_size`, `seq_len`, `n_cat_features`)
  - `x_seq_num_cid`: Numerical sequence features for cid sequence of shape (`batch_size`, `seq_len`, `n_num_features`)
  - `time_seq_cid`: Time delta sequence keyed by cid
  - `x_seq_cat_ccid`: Categorical sequence features for ccid sequence of shape (`batch_size`, `seq_len`, `n_cat_features`)
  - `x_seq_num_ccid`: Numerical sequence features for ccid sequence of shape (`batch_size`, `seq_len`, `n_num_features`)
  - `time_seq_ccid`: Time delta sequence keyed by ccid
  - `x_engineered`: Numerical engineered features of shape (`batch_size`, `dim_embedding_table`)
  - `attn_mask`: Attention mask (optional)
  - `key_padding_mask_cid`: Key padding mask (optional) for cid sequence
  - `key_padding_mask_ccid`: Key padding mask (optional) for ccid sequence
- Outputs:
  - `scores`: Output class scores, shape (`batch_size`, `n_classes`)
  - `ensemble`: Ensemble embeddings, shape (`batch_size`, `3 * dim_embedding_table` (+ `dim_embedding_table` if use_mlp))
- Parameters:
  - `n_cat_features`: Number of categorical sequence features 
  - `n_num_features`: Number of numerical sequence features 
  - `n_classes`: Number of output classes
  - `n_embedding`: Size of sequence embedding table
  - `seq_len`: int - Sequence length
  - `n_engineered_num_features`: int - Number of numerical engineered features 
  - `dim_embedding_table`: int - Dimension of embedding table
  - `dim_attn_feedforward`: Dimension of feedforward network inside transformer layer
  - `num_heads`: Number of attention heads
  - `dropout`: Dropout probability (default 0.1)
  - `n_layers_order`: Number of order attention layers
  - `n_layers_feature`: Number of feature attention layers
  - `emb_tbl_use_bias`: Whether to use bias term to embeddings
  - `use_moe`: Whether to use mixture-of-experts (MoE) structure for transformer layer
  - `num_experts`: Number of experts to use for MoE layer
  - `use_time_seq`: Whether to use time information
  - `return_seq`: Whether to return sequence of embeddings
- Forward Pass
  - Calculate gate embeddings for each sequence
  - Get gate score for each sequence with gate embeddings
  - Apply order attention for cid and/or ccid sequences (skip ccid if gate score is small)
  - Apply feature attention 
  - Combine embeddings from cid/ccid sequences with gate scores
  - Concatenate order and feature embeddings
  - Feed concatenated embeddings to classifier to produce scores

### `MODS/tsa/scripts/basic_blocks.py`
#### OrderAttentionLayer
- This layer takes singe sequence data, expand one dimension with trainable embedding table, and aggregate on the feature level before feeding to multiple `AttentionLayer`. Currently feature aggregation method is an MLP like encoder.
- Inputs:
  - `x_cat`: Categorical sequence features of shape (`batch_size`, `seq_len`, `n_cat_features`)
  - `x_num`: Numerical sequence features of shape (`batch_size`, `seq_len`, `n_num_features`)
  - `time_seq`: Time delta sequence
  - `attn_mask`: Attention mask (optional)
  - `key_padding_mask`: Key padding mask (optional)
- Outputs:
  - `x`: Output tensor of shape (`batch_size`, `dim_embed`) or (`batch_size`, `seq_len`, `dim_embed`)
- Parameters:
  - `n_cat_features`: Number of categorical sequence features 
  - `n_num_features`: Number of numerical sequence features
  - `n_embedding`: Size of sequence embedding table
  - `seq_len`: Sequence length
  - `dim_embed`: Output embedding dimension, equalling to 2 * Embedding dimension of embedding table
  - `dim_attn_feedforward`: Dimension of feedforward network inside `AttentionLayer`
  - `embedding_table`: Embedding table for sequence features
  - `num_heads`: Number of attention heads
  - `dropout`: Dropout rate
  - `n_layers_order`: Number of attention layers
  - `emb_tbl_use_bias`: Whether to use bias term to embeddings
  - `use_moe`: Whether to use mixture-of-experts (MoE) structure for transformer layer
  - `num_experts`: Number of experts to use for MoE layer
  - `use_time_seq`: Whether to use time information
  - `return_seq`: Whether to return sequence of embeddings for the output
  
#### FeatureAttentionLayer
- This layer takes singe sequence data, expand one dimension with trainable embedding table, and aggregate on the order level before feeding to multiple `AttentionLayerPreNorm`. Currently order aggregation method is to take the last order.
- Inputs:
  - `x_cat`: Categorical sequence features of shape (`batch_size`, `seq_len`,  `n_cat_features`)
  - `x_num`: Numerical sequence features of shape (`batch_size`, `seq_len`, `n_num_features`)
  - `x_engineered`: Numerical engineered features of shape (`batch_size`, `dim_embedding_table`)
- Outputs:
  - `x`: Output tensor of shape (`batch_size`, `dim_embed`)
- Parameters:
  - `n_cat_features`: Number of categorical sequence features 
  - `n_num_features`: Number of numerical sequence features
  - `n_embedding`: Size of sequence embedding table
  - `n_engineered_num_features`: int - Number of numerical engineered features
  - `dim_embed`: Output embedding dimension, equalling to 2 * Embedding dimension of embedding table
  - `dim_attn_feedforward`: Dimension of feedforward network inside `AttentionLayerPreNorm`
  - `embedding_table`: Embedding table for sequence features
  - `embedding_table_engineered`: Embedding table for engineered features
  - `num_heads`: Number of attention heads
  - `dropout`: Dropout rate
  - `n_layers_feature`: Number of attention layers
  - `emb_tbl_use_bias`: Whether to use bias term to embeddings
  - `use_moe`: Whether to use mixture-of-experts (MoE) structure for transformer layer
  - `num_experts`: Number of experts to use for MoE layer

#### AttentionLayer
- A simple attention layer using `TemporalMultiheadAttention`.
- Inputs:
  - `x`: Input tensor of shape (`seq_len`, `batch_size`, `dim_embed`) 
  - `time`: Time delta tensor
  - `attn_mask`: Attention mask (optional)
  - `key_padding_mask`: Key padding mask (optional)
- Outputs:
  - `x`: Output tensor of shape (`seq_len`, `batch_size`,  `dim_embed`) after applying self-attention 
  
#### AttentionLayerPreNorm
- Attention layer with pre-normalization.
- Inputs:
  - `x`: Input tensor of shape (`seq_len`, `batch_size`, `dim_embed/2`) 
  - `attn_mask`: Attention mask (optional)
  - `key_padding_mask`: Key padding mask (optional)
- Outputs:
  - `x`: Output tensor of shape (`seq_len`, `batch_size`, `dim_embed/2`) after applying self-attention

#### FeatureAggregation
- Encodes input features using a multi-layer perceptron. 
- Inputs:
  - `x`: Input features tensor of shape (`batch_size`, `seq_len`, `num_features`) 
- Outputs:
  - `encode`: Encoded features tensor of shape (`batch_size`, `seq_len`, 1)

#### `TimeEncode`/`TimeEncoder`:
- A time encoding layer that encodes time information with periodic fuction sinusoids. Takes in time values and outputs encoded time embeddings.

#### `compute_FM_parallel`
- Computes second order feature interactions using Factorization Machine. Takes in a tensor of feature embeddings and outputs FM values. Can be used as feature aggregation function.

## Reference:
- Xueyu Mao, Jia Geng, Olcay Boz, Qi Zhao, Lei Shi, Zora Zhang, Zhanlong Qiu, [On Temporal Self-Attention and Auto Feature Generation](https://amazon.awsapps.com/workdocs-preview/index.html#/document/1382fd99a497c485b5268373d099dd9bea823fef342bf69019b04ac9bb03f8e0), AMLC 2023
- Xueyu Mao, [TSA: Temporal Self-Attention for auto feature generation and beyond](https://amazon.awsapps.com/workdocs-preview/index.html#/document/6a58610384d58642c792887c85d5c29734b1ec155ddd76e45b20e0750f54fd74), [SPS ML Bi-Weekly Presentation](https://amazon.awsapps.com/workdocs-preview/index.html#/document/fcc8218568acc1bbe51dff634530131d66425fb760c61c453b9186057661aaec), June 16, 2023.
- Develop repository: https://code.amazon.com/packages/Sequence-Modeling/trees/mainline

## Production Code:
- [EUTSASuspectQueueModel](https://tiny.amazon.com/1i3km79oa/PaymentRiskModsTemplate/mainline/eu_tsa_sq)
- [NATSASuspectQueueModel](https://tiny.amazon.com/v72ir830/PaymentRiskModsTemplate/3f9e2822/na_tsa_sq)

## Continual Work
- Multi Task: [MT-TSA](https://code.amazon.com/packages/BRPMultiTaskTSA/trees/mainline) 
- Self-supervised pre-training: [SeqGPT](https://code.amazon.com/packages/Seqgpt/trees/mainline) &rarr; [SandStone](https://code.amazon.com/packages/SandstoneFBMTemplate/trees/mainline)