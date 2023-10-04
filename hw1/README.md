Configuration saved in /tmp2/b10902069/adl_hw1/config.json
Model weights saved in /tmp2/b10902069/adl_hw1/pytorch_model.bin
tokenizer config file saved in /tmp2/b10902069/adl_hw1/tokenizer_config.json
Special tokens file saved in /tmp2/b10902069/adl_hw1/special_tokens_map.json

Configuration saved in /tmp2/b10902069/adl_hw1/config.json
Model weights saved in /tmp2/b10902069/adl_hw1/pytorch_model.bin
tokenizer config file saved in /tmp2/b10902069/adl_hw1/tokenizer_config.json
Special tokens file saved in /tmp2/b10902069/adl_hw1/special_tokens_map.json

b10902069@meow1 [~/workspace/ADL/hw1] ./train.sh extractive
2023-10-04 23:34:57.290923: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/usr/lib/python3.11/site-packages/h5py/__init__.py:36: UserWarning: h5py is running against HDF5 1.14.2 when it was built against 1.14.1, this may cause problems
  _warn(("h5py is running against HDF5 {0} when it was built against {1}, "
10/04/2023 23:34:59 - INFO - __main__ - Distributed environment: DistributedType.NO
Num processes: 1
Process index: 0
Local process index: 0
Device: cuda

Mixed precision type: no

Downloading data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 14716.86it/s]
Extracting data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 196.30it/s]
Generating train split: 21714 examples [00:00, 54690.05 examples/s]
Generating validation split: 3009 examples [00:00, 58119.55 examples/s]
loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading file vocab.txt from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/vocab.txt
loading file tokenizer.json from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/tokenizer.json
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/tokenizer_config.json
loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading weights file model.safetensors from cache at /tmp2/b10902069/adl_hw1/extractive_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/model.safetensors
Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForQuestionAnswering: ['cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running tokenizer on train dataset: 100%|███████████████████████████████████████████████████████████████████| 21714/21714 [00:13<00:00, 1627.36 examples/s]
Running tokenizer on validation dataset: 100%|█████████████████████████████████████████████████████████████████| 3009/3009 [00:03<00:00, 890.28 examples/s]
10/04/2023 23:35:19 - INFO - __main__ - Sample 9538 of the training set: {'input_ids': [101, 1751, 2157, 1295, 3298, 1762, 10058, 8158, 2399, 722, 2527, 6158, 4493, 7938, 5175, 5251, 1357, 807, 136, 102, 1751, 2157, 5474, 4673, 1762, 10058, 8158, 2399, 1357, 807, 749, 1751, 2157, 1295, 3298, 8024, 699, 1146, 6981, 5645, 2124, 4638, 1061, 1399, 2768, 1519, 2201, 2253, 1765, 1818, 8024, 1920, 6205, 3817, 7386, 1728, 4158, 1469, 5153, 5147, 1398, 1962, 7386, 7028, 4540, 5445, 6158, 7370, 1399, 8024, 679, 719, 2527, 1398, 1962, 7386, 738, 6158, 7705, 6852, 8024, 1506, 4294, 4886, 3918, 5965, 7386, 1357, 5445, 807, 722, 699, 3291, 1399, 4158, 2357, 7798, 1046, 3360, 1506, 4294, 4886, 7386, 511, 800, 947, 1762, 10058, 8161, 2399, 886, 4500, 5474, 6930, 4413, 1842, 976, 4158, 712, 1842, 4684, 1168, 6237, 3141, 511, 2527, 889, 2768, 4158, 1751, 2157, 5474, 4673, 6887, 1936, 7386, 4638, 6929, 943, 2357, 7798, 1046, 3360, 3472, 4413, 936, 3556, 6956, 2768, 4989, 3176, 9460, 8152, 2399, 8024, 699, 1762, 7392, 2399, 1217, 1057, 5401, 1751, 1295, 3298, 511, 3634, 7386, 1333, 6158, 4935, 4158, 1920, 6205, 3817, 7386, 8024, 2527, 4935, 4158, 4129, 782, 7386, 511, 10988, 2399, 3149, 1399, 4413, 1519, 2970, 6865, 5178, 2042, 2527, 8024, 2054, 7768, 7274, 1993, 2200, 3634, 7386, 4935, 4158, 2357, 7798, 1046, 3360, 3173, 6947, 7386, 8024, 3173, 6947, 7386, 6560, 2533, 12116, 2399, 4638, 5401, 5474, 1094, 6725, 511, 11787, 2399, 6752, 1168, 1751, 2157, 5474, 4673, 2527, 8024, 3634, 7386, 2768, 4158, 6865, 5265, 6560, 2533, 712, 6206, 5474, 4673, 679, 1398, 5474, 6555, 4638, 676, 3118, 4413, 7386, 704, 4638, 5018, 671, 3118, 8024, 738, 3221, 1071, 704, 1546, 671, 671, 3118, 3472, 4413, 7386, 511, 6857, 722, 2527, 6882, 749, 1061, 2399, 8024, 2798, 1086, 3613, 1357, 2533, 2768, 1216, 8024, 1728, 4158, 1060, 936, 3556, 6956, 1066, 6536, 8024, 2571, 6237, 3141, 4638, 1751, 5474, 2349, 4273, 4638, 3040, 7032, 7873, 7386, 6546, 749, 3149, 855, 1399, 782, 1828, 677, 4638, 4413, 1519, 5645, 2357, 7798, 1046, 3360, 7386, 8024, 671, 6629, 6882, 889, 4638, 6917, 3300, 5244, 3136, 5230, 1298, 2548, 185, 4031, 7384, 8024, 6857, 886, 2357, 7798, 1046, 3360, 7386, 4989, 1174, 7379, 1057, 1058, 4261, 511, 2357, 7798, 1046, 3360, 6631, 7464, 7386, 7386, 2527, 679, 6511, 2346, 1399, 8024, 6560, 2533, 749, 11886, 2399, 1469, 8985, 2399, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'start_positions': 20, 'end_positions': 23}.
10/04/2023 23:35:19 - INFO - __main__ - Sample 36621 of the training set: {'input_ids': [101, 4507, 1506, 5164, 185, 7681, 185, 3294, 7351, 5838, 4273, 2990, 6359, 1462, 1399, 4638, 519, 5474, 6930, 1751, 7344, 6725, 520, 1762, 1525, 671, 2399, 3229, 3633, 2466, 2768, 4989, 136, 102, 5632, 4507, 3696, 712, 7955, 1139, 6716, 4638, 1184, 2200, 7526, 1506, 5164, 185, 7681, 185, 3294, 7351, 5838, 4273, 2990, 6359, 2828, 3173, 2456, 4638, 6725, 7386, 1462, 1399, 4158, 519, 5474, 6930, 1751, 7344, 6725, 520, 8024, 4363, 2533, 6205, 2548, 5474, 6930, 6359, 3298, 4638, 6291, 1377, 511, 9043, 2399, 8111, 3299, 8110, 3189, 8024, 5474, 6930, 1751, 7344, 6725, 3633, 2466, 2456, 4989, 8024, 6283, 1921, 3221, 3249, 7798, 1894, 2200, 7526, 3419, 1506, 2548, 185, 7681, 185, 3763, 2617, 7452, 3172, 4294, 6293, 6801, 1060, 4636, 1453, 2399, 511, 1762, 517, 2548, 2692, 2562, 5474, 6930, 1066, 1469, 1751, 1825, 3315, 3791, 518, 4638, 934, 3633, 3428, 2527, 8024, 9043, 2399, 6205, 2548, 3633, 2466, 1217, 1057, 1266, 5147, 8024, 9076, 2399, 7274, 1993, 2972, 6121, 2547, 1070, 1169, 8024, 2399, 2234, 8123, 5635, 8208, 3641, 4638, 1059, 1751, 4511, 2595, 2553, 7519, 5412, 1243, 3302, 1070, 2514, 511, 1398, 3229, 4158, 7547, 1350, 679, 7544, 2537, 752, 3636, 6172, 1249, 1243, 5442, 4638, 1825, 3315, 782, 3609, 8024, 2548, 1751, 1825, 3315, 3791, 699, 6548, 750, 782, 3696, 2867, 3302, 1070, 2514, 4638, 3609, 1164, 8024, 1006, 7444, 1403, 2514, 3124, 3582, 7302, 4633, 6250, 1315, 1377, 3121, 3302, 3296, 807, 4638, 4852, 3298, 2514, 511, 5632, 8163, 2399, 128, 3299, 7274, 1993, 8024, 2548, 1751, 2347, 1059, 7481, 2450, 3632, 2547, 1070, 1350, 1392, 4934, 3296, 807, 2514, 511, 1107, 2782, 3309, 7279, 8024, 6205, 2548, 5474, 6930, 1751, 7344, 6725, 2768, 4158, 1266, 5147, 1762, 704, 3627, 4638, 7344, 6127, 712, 1213, 511, 4534, 3229, 4638, 6205, 2548, 5474, 6930, 1751, 7344, 6725, 3075, 3300, 8249, 5857, 126, 1283, 782, 4638, 1070, 1213, 8024, 809, 1350, 8126, 5857, 4638, 7478, 4412, 2514, 4638, 3152, 5480, 782, 1519, 511, 7380, 6725, 676, 943, 6725, 4507, 8124, 943, 2374, 2792, 5175, 2768, 8024, 1788, 1046, 1350, 6172, 4508, 6880, 1070, 6722, 738, 5195, 6882, 3636, 6172, 8039, 4958, 6725, 1920, 6956, 819, 4638, 2782, 7784, 3582, 2768, 4158, 1266, 5147, 5474, 1394, 7344, 4958, 6725, 4638, 671, 6956, 819, 8039, 3862, 6725, 1179, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'start_positions': 85, 'end_positions': 86}.
10/04/2023 23:35:19 - INFO - __main__ - Sample 4075 of the training set: {'input_ids': [101, 5101, 6832, 765, 3274, 7983, 7386, 6237, 3141, 4638, 1333, 1728, 3221, 136, 102, 1519, 6626, 5401, 2802, 2874, 8024, 4680, 1184, 2347, 3300, 7376, 7032, 7081, 510, 3293, 7093, 6740, 510, 4374, 2456, 3696, 510, 6958, 3790, 2562, 510, 5529, 7032, 7983, 510, 960, 4886, 2548, 510, 7376, 971, 3668, 510, 3360, 1528, 4440, 510, 3330, 2920, 3208, 510, 5397, 1649, 785, 510, 4374, 5204, 704, 510, 3360, 2094, 971, 5023, 782, 3295, 5195, 4633, 677, 1920, 5474, 4673, 8024, 699, 2898, 5265, 3300, 5865, 679, 7097, 4638, 2768, 5245, 511, 8127, 2399, 125, 3299, 8108, 3189, 8024, 4255, 5101, 5709, 1909, 2108, 3472, 4413, 5474, 4673, 2146, 2357, 2768, 4989, 511, 5637, 4124, 5091, 4413, 1762, 8447, 172, 8338, 2399, 7279, 3295, 5195, 3300, 704, 5836, 5480, 3511, 5091, 4413, 5474, 4673, 8024, 2527, 889, 4028, 6365, 4158, 6631, 5159, 5091, 4413, 5474, 6555, 8024, 4412, 3300, 5637, 1266, 6888, 3615, 2339, 4923, 510, 3425, 1754, 4468, 1754, 2456, 5064, 510, 1378, 4124, 1566, 6983, 510, 2168, 6930, 1235, 1894, 510, 7032, 7271, 6983, 2449, 510, 5637, 4124, 7065, 6121, 1469, 6168, 7384, 5152, 3255, 2949, 5023, 673, 7386, 511, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'start_positions': 0, 'end_positions': 0}.
/tmp2/b10902069/python_package/accelerate/accelerator.py:523: FutureWarning: The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use `Accelerator.mixed_precision == 'fp16'` instead.
  warnings.warn(
Downloading builder script: 100%|█████████████████████████████████████████████████████████████████████████████████████| 4.53k/4.53k [00:00<00:00, 9.43MB/s]
Downloading extra modules: 100%|██████████████████████████████████████████████████████████████████████████████████████| 3.32k/3.32k [00:00<00:00, 8.53MB/s]
The device is using:  cuda
10/04/2023 23:35:23 - INFO - __main__ - ***** Running training *****
10/04/2023 23:35:23 - INFO - __main__ -   Num examples = 38143
10/04/2023 23:35:23 - INFO - __main__ -   Num Epochs = 1
10/04/2023 23:35:23 - INFO - __main__ -   Instantaneous batch size per device = 16
10/04/2023 23:35:23 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 16
10/04/2023 23:35:23 - INFO - __main__ -   Gradient Accumulation steps = 1
10/04/2023 23:35:23 - INFO - __main__ -   Total optimization steps = 2384
  0%|                                                                                                                             | 0/2384 [00:00<?, ?it/s]You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2384/2384 [15:05<00:00,  2.75it/s]10/04/2023 23:50:28 - INFO - __main__ - ***** Running Evaluation *****
10/04/2023 23:50:28 - INFO - __main__ -   Num examples = 5503
10/04/2023 23:50:28 - INFO - __main__ -   Batch size = 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3009/3009 [00:15<00:00, 200.22it/s]
10/04/2023 23:51:46 - INFO - __main__ - Evaluation metrics: {'exact_match': 78.39813891658358, 'f1': 78.39813891658358} 2995/3009 [00:14<00:00, 199.67it/s]
Configuration saved in /tmp2/b10902069/adl_hw1/extractive_dir/config.json
Model weights saved in /tmp2/b10902069/adl_hw1/extractive_dir/pytorch_model.bin
tokenizer config file saved in /tmp2/b10902069/adl_hw1/extractive_dir/tokenizer_config.json
Special tokens file saved in /tmp2/b10902069/adl_hw1/extractive_dir/special_tokens_map.json
10/04/2023 23:51:47 - INFO - __main__ - {
    "exact_match": 78.39813891658358,
    "f1": 78.39813891658358
}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2384/2384 [16:24<00:00,  2.42it/s]

b10902069@meow1 [~/workspace/ADL/hw1] ./train.sh multiple_choice
2023-10-04 23:15:49.669328: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
/usr/lib/python3.11/site-packages/h5py/__init__.py:36: UserWarning: h5py is running against HDF5 1.14.2 when it was built against 1.14.1, this may cause problems
  _warn(("h5py is running against HDF5 {0} when it was built against {1}, "
10/04/2023 23:15:52 - INFO - __main__ - Distributed environment: DistributedType.NO
Num processes: 1
Process index: 0
Local process index: 0
Device: cuda

Mixed precision type: no

loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading file vocab.txt from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/vocab.txt
loading file tokenizer.json from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/tokenizer.json
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/tokenizer_config.json
loading configuration file config.json from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/config.json
Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

loading weights file model.safetensors from cache at /tmp2/b10902069/adl_hw1/multiple_choice_dir/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/model.safetensors
Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForMultipleChoice: ['cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMultipleChoice from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMultipleChoice from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForMultipleChoice were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 21714/21714 [00:21<00:00, 1020.02 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3009/3009 [00:02<00:00, 1064.84 examples/s]
10/04/2023 23:16:19 - INFO - __main__ - Sample 4769 of the training set: {'input_ids': [[101, 704, 1751, 2970, 3119, 7676, 3949, 782, 4638, 784, 7938, 3346, 6205, 886, 2533, 7676, 3949, 782, 1927, 1343, 928, 2552, 102, 4412, 807, 3229, 3309, 4638, 7274, 1993, 807, 6134, 749, 3627, 3828, 3152, 5971, 2541, 5646, 4638, 5178, 3338, 511, 4825, 1147, 4638, 3229, 7279, 7953, 3298, 898, 3087, 1392, 943, 7526, 1818, 4638, 4518, 2137, 5445, 3300, 2792, 679, 1398, 8024, 3683, 1963, 6303, 8024, 671, 943, 3644, 1380, 2119, 2157, 1377, 5543, 3298, 2200, 4412, 807, 4638, 7274, 1993, 3123, 1762, 9316, 8129, 2399, 8024, 5445, 671, 943, 7509, 3556, 2157, 1179, 1377, 5543, 3298, 2828, 3229, 7279, 3123, 1762, 7509, 3556, 4638, 3857, 4035, 3229, 3309, 5178, 3338, 722, 2527, 8024, 738, 2218, 3221, 1920, 5147, 8985, 2399, 511, 4534, 807, 102], [101, 704, 1751, 2970, 3119, 7676, 3949, 782, 4638, 784, 7938, 3346, 6205, 886, 2533, 7676, 3949, 782, 1927, 1343, 928, 2552, 102, 699, 924, 6349, 7676, 3949, 7370, 1912, 769, 1350, 7344, 1243, 1912, 3176, 1071, 2124, 752, 1243, 775, 3300, 7770, 2428, 5632, 3780, 3609, 511, 1358, 1168, 1063, 1724, 752, 816, 2512, 7513, 8024, 7676, 3949, 782, 2205, 712, 3609, 4919, 769, 1469, 704, 1751, 3124, 2424, 3291, 1217, 1927, 1343, 928, 2552, 8024, 782, 2552, 2684, 2684, 8024, 7676, 3949, 4638, 6536, 4496, 1019, 3419, 678, 6649, 8024, 4919, 3696, 4060, 1086, 3613, 1139, 4412, 8024, 7676, 3949, 3124, 2424, 6876, 2972, 1139, 7676, 3949, 3582, 1842, 3417, 2552, 6243, 1205, 809, 808, 1062, 3696, 4952, 2137, 511, 704, 1751, 5018, 673, 2234, 102], [101, 704, 1751, 2970, 3119, 7676, 3949, 782, 4638, 784, 7938, 3346, 6205, 886, 2533, 7676, 3949, 782, 1927, 1343, 928, 2552, 102, 8464, 2399, 807, 1159, 3309, 8024, 4507, 3176, 3173, 4518, 1759, 1765, 1943, 5147, 1558, 7539, 8024, 886, 3658, 3696, 1765, 3124, 2424, 6206, 5440, 2719, 7676, 3949, 1184, 6854, 1558, 7539, 511, 5739, 1751, 3124, 2424, 3295, 1914, 3613, 6206, 3724, 704, 1751, 4534, 2229, 2454, 5265, 3173, 4518, 4909, 5147, 511, 8499, 2399, 807, 1159, 8024, 5739, 1751, 2990, 1139, 1146, 2858, 7676, 3949, 712, 3609, 1469, 3780, 3609, 8024, 712, 3609, 3645, 704, 1751, 8024, 5739, 1751, 793, 924, 4522, 5052, 3780, 3609, 8024, 1772, 6158, 704, 1066, 2867, 5179, 511, 8499, 2399, 807, 1159, 8024, 3173, 4518, 4909, 5147, 2200, 102], [101, 704, 1751, 2970, 3119, 7676, 3949, 782, 4638, 784, 7938, 3346, 6205, 886, 2533, 7676, 3949, 782, 1927, 1343, 928, 2552, 102, 1154, 2180, 4638, 7679, 1046, 2590, 712, 5412, 928, 2573, 6291, 4158, 679, 5543, 4684, 2970, 2200, 4534, 1184, 4638, 1751, 2157, 3121, 6863, 4158, 1066, 4496, 712, 5412, 4852, 3298, 8024, 5445, 2553, 7519, 7674, 1044, 6868, 1057, 4852, 3298, 712, 5412, 3229, 3309, 8024, 1728, 3634, 800, 712, 6206, 7302, 2552, 1963, 862, 2200, 915, 5397, 3172, 6752, 6365, 4158, 4852, 3298, 712, 5412, 4852, 3298, 511, 4158, 749, 5543, 1917, 6857, 3564, 976, 8024, 800, 6291, 4158, 4192, 4496, 7389, 5159, 2201, 3124, 3221, 2553, 6206, 4638, 8024, 809, 1886, 1169, 6536, 4496, 7389, 5159, 1469, 4634, 2245, 4852, 3298, 712, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': 1}.
10/04/2023 23:16:19 - INFO - __main__ - Sample 18310 of the training set: {'input_ids': [[101, 784, 7938, 712, 5412, 3221, 6158, 6291, 4158, 3300, 3136, 7621, 4638, 2900, 2206, 3136, 5509, 102, 1762, 1392, 943, 3124, 3609, 4948, 3513, 1217, 2485, 1398, 1265, 3124, 5032, 4638, 2512, 7513, 1213, 722, 678, 8024, 1333, 857, 3696, 3184, 2823, 1168, 749, 1469, 1071, 800, 3152, 1265, 3291, 1920, 4638, 2970, 6240, 3582, 3298, 8024, 2215, 1071, 3221, 2398, 1815, 3184, 511, 6857, 4934, 1398, 1265, 1469, 6900, 2746, 4638, 6882, 4923, 3300, 3229, 3298, 2227, 7401, 5865, 2451, 3793, 4638, 4852, 3298, 4060, 3837, 8024, 2215, 1071, 3221, 4934, 3184, 3560, 6250, 4638, 6365, 6907, 8024, 5445, 684, 6857, 763, 6882, 4923, 738, 1762, 809, 1184, 4638, 5637, 4124, 3300, 5865, 6352, 1162, 3184, 5408, 4638, 1216, 5543, 1762, 511, 5445, 6857, 4934, 102], [101, 784, 7938, 712, 5412, 3221, 6158, 6291, 4158, 3300, 3136, 7621, 4638, 2900, 2206, 3136, 5509, 102, 5637, 4124, 1392, 1333, 857, 3696, 3184, 3075, 3300, 1392, 5632, 4638, 6629, 3975, 1001, 6303, 8024, 6818, 2399, 889, 898, 3087, 6295, 6241, 2119, 510, 5440, 1367, 2119, 1469, 3152, 1265, 782, 7546, 2119, 5023, 4638, 4777, 4955, 2972, 3174, 8024, 1762, 1282, 673, 686, 5145, 4031, 782, 4919, 3696, 5637, 4124, 722, 1184, 8024, 5637, 4124, 1333, 857, 3696, 3184, 1762, 5637, 4124, 4638, 3833, 1240, 2347, 3300, 1920, 5147, 1061, 1283, 2399, 722, 719, 511, 5637, 4124, 1333, 857, 3696, 3184, 1762, 6909, 1001, 2119, 1469, 6295, 6241, 2119, 4638, 1146, 7546, 677, 2253, 3176, 1298, 2294, 3696, 3184, 1469, 1298, 2294, 6295, 5143, 8024, 1469, 102], [101, 784, 7938, 712, 5412, 3221, 6158, 6291, 4158, 3300, 3136, 7621, 4638, 2900, 2206, 3136, 5509, 102, 7376, 3717, 2793, 5000, 6908, 5244, 5186, 3229, 3295, 5645, 1392, 3184, 1333, 857, 3696, 807, 6134, 4634, 6134, 1333, 857, 3696, 3184, 5645, 5637, 4124, 3173, 3124, 2424, 3173, 4638, 1919, 845, 7302, 913, 3454, 5147, 8024, 886, 2533, 1333, 857, 3696, 3184, 5632, 3780, 3176, 809, 2527, 2768, 4158, 3696, 6868, 7955, 3124, 2424, 6908, 5647, 3229, 4638, 712, 6206, 1366, 5998, 511, 704, 5836, 3696, 1751, 2740, 3791, 771, 3176, 5018, 1063, 3613, 1872, 934, 3229, 8024, 3176, 1872, 934, 5018, 1282, 3454, 1825, 3315, 1751, 5032, 704, 8024, 3633, 2466, 2824, 6291, 1333, 857, 3696, 3184, 4638, 3696, 3184, 3609, 8024, 3176, 5018, 1282, 671, 102], [101, 784, 7938, 712, 5412, 3221, 6158, 6291, 4158, 3300, 3136, 7621, 4638, 2900, 2206, 3136, 5509, 102, 1963, 1398, 3680, 4934, 7546, 1798, 4638, 3124, 3780, 2590, 2682, 8024, 1762, 2179, 7396, 677, 8024, 1825, 4719, 3136, 3696, 712, 712, 5412, 2792, 712, 2484, 4638, 3124, 5032, 1469, 7028, 2552, 1762, 679, 1398, 4638, 3229, 3309, 1469, 679, 1398, 4638, 1751, 2157, 6174, 738, 3300, 6258, 1914, 2345, 4530, 511, 1825, 4719, 3136, 3696, 712, 712, 5412, 6858, 2382, 2253, 3176, 4852, 3298, 924, 2127, 712, 5412, 8024, 738, 1728, 3634, 6258, 1914, 1825, 4719, 3136, 3696, 712, 712, 5412, 5442, 1353, 2205, 1876, 5522, 1469, 1398, 2595, 2042, 2012, 8024, 679, 6882, 3378, 763, 1825, 4719, 3136, 3696, 712, 712, 5412, 3124, 7955, 793, 3298, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': 0}.
10/04/2023 23:16:19 - INFO - __main__ - Sample 2037 of the training set: {'input_ids': [[101, 1751, 3696, 7955, 704, 1925, 5018, 753, 2234, 2382, 1999, 3298, 5018, 673, 3613, 3097, 1920, 3298, 6359, 3221, 1762, 862, 5993, 5647, 6121, 136, 102, 124, 3299, 8113, 3189, 8024, 704, 6662, 6725, 4638, 4923, 4051, 1462, 1259, 1752, 1298, 776, 4638, 1392, 6956, 7386, 1403, 1298, 776, 6867, 6920, 1392, 3087, 7953, 4634, 6629, 4338, 3122, 8024, 5195, 6882, 124, 3189, 4080, 2782, 8024, 7380, 5265, 3122, 1046, 7378, 7120, 510, 4910, 7377, 7302, 510, 7983, 6963, 7120, 510, 3736, 2180, 7120, 8024, 4684, 6873, 1298, 776, 6818, 6946, 511, 862, 2746, 3620, 2792, 4372, 3346, 6662, 6725, 738, 1168, 6888, 7120, 3736, 511, 124, 3299, 8133, 3189, 8024, 4923, 4051, 678, 808, 5244, 3122, 1298, 776, 8024, 4534, 1921, 1315, 4960, 1057, 1298, 102], [101, 1751, 3696, 7955, 704, 1925, 5018, 753, 2234, 2382, 1999, 3298, 5018, 673, 3613, 3097, 1920, 3298, 6359, 3221, 1762, 862, 5993, 5647, 6121, 136, 102, 2834, 3189, 2782, 4261, 3309, 7279, 8024, 3085, 818, 1751, 6725, 3124, 3780, 6956, 1199, 712, 818, 1076, 5018, 1061, 6662, 6725, 7688, 3939, 6794, 752, 5993, 712, 818, 8024, 6511, 6519, 1751, 1066, 7427, 3175, 6725, 752, 5645, 3124, 3780, 6310, 977, 511, 3189, 3315, 2832, 7360, 2527, 8024, 7373, 1398, 704, 1066, 704, 1925, 1999, 1519, 3298, 712, 2375, 3688, 4075, 3346, 1184, 2518, 7028, 2723, 8024, 5645, 704, 1751, 1751, 3696, 7955, 5244, 6161, 5919, 704, 3633, 3176, 7028, 2723, 6312, 1161, 511, 5018, 753, 3613, 1751, 1066, 1058, 2782, 3309, 7279, 8024, 3085, 818, 704, 1925, 102], [101, 1751, 3696, 7955, 704, 1925, 5018, 753, 2234, 2382, 1999, 3298, 5018, 673, 3613, 3097, 1920, 3298, 6359, 3221, 1762, 862, 5993, 5647, 6121, 136, 102, 7481, 2205, 704, 1751, 1751, 3696, 7955, 1058, 6956, 4757, 4688, 1469, 5160, 4261, 8024, 809, 1350, 4852, 3298, 6748, 6316, 2485, 4164, 6206, 3724, 3696, 712, 510, 3791, 3780, 8024, 5919, 704, 3633, 1762, 704, 1751, 1751, 3696, 7955, 676, 2234, 1724, 704, 1059, 3298, 677, 8024, 712, 2484, 1374, 7274, 1751, 3696, 3298, 6359, 510, 1169, 2137, 519, 6246, 3124, 3229, 3309, 5147, 3791, 520, 8024, 5529, 4031, 3696, 1353, 2205, 8039, 5919, 3176, 9639, 2399, 123, 3299, 8143, 3189, 2200, 5529, 4031, 3696, 6727, 4881, 3176, 1298, 776, 3966, 2255, 8024, 2206, 5636, 126, 3299, 2451, 2336, 102], [101, 1751, 3696, 7955, 704, 1925, 5018, 753, 2234, 2382, 1999, 3298, 5018, 673, 3613, 3097, 1920, 3298, 6359, 3221, 1762, 862, 5993, 5647, 6121, 136, 102, 3418, 3087, 517, 704, 5836, 782, 3696, 1066, 1469, 1751, 2740, 3791, 518, 6211, 2137, 8024, 1059, 1751, 782, 3696, 807, 6134, 1920, 3298, 678, 6257, 5735, 2397, 2201, 7271, 1999, 1519, 3298, 511, 7370, 749, 1762, 3152, 1265, 1920, 7484, 1462, 3309, 7279, 1374, 7274, 4638, 5018, 1724, 2234, 1059, 1751, 782, 1920, 3760, 3300, 6257, 4989, 1912, 8024, 1071, 800, 3644, 2234, 782, 1920, 3298, 6359, 6963, 6257, 3300, 2201, 7271, 1999, 1519, 3298, 511, 1071, 704, 8024, 5018, 671, 5635, 758, 2234, 6257, 3300, 3696, 3184, 510, 3791, 3428, 510, 7521, 5050, 510, 807, 6134, 6536, 3419, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': 0}.
/tmp2/b10902069/python_package/accelerate/accelerator.py:523: FutureWarning: The `use_fp16` property is deprecated and will be removed in version 1.0 of Accelerate use `Accelerator.mixed_precision == 'fp16'` instead.
  warnings.warn(
The device is using:  cuda
10/04/2023 23:16:21 - INFO - __main__ - ***** Running training *****
10/04/2023 23:16:21 - INFO - __main__ -   Num examples = 21714
10/04/2023 23:16:21 - INFO - __main__ -   Num Epochs = 1
10/04/2023 23:16:21 - INFO - __main__ -   Instantaneous batch size per device = 16
10/04/2023 23:16:21 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 16
10/04/2023 23:16:21 - INFO - __main__ -   Gradient Accumulation steps = 1
10/04/2023 23:16:21 - INFO - __main__ -   Total optimization steps = 1358
  0%|                                                                                                                             | 0/1358 [00:00<?, ?it/s]You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉| 1357/1358 [09:30<00:00,  2.23it/s]epoch 0: {'accuracy': 0.7833167165171153}
Configuration saved in /tmp2/b10902069/adl_hw1/multiple_choice_dir/config.json
Model weights saved in /tmp2/b10902069/adl_hw1/multiple_choice_dir/pytorch_model.bin
tokenizer config file saved in /tmp2/b10902069/adl_hw1/multiple_choice_dir/tokenizer_config.json
Special tokens file saved in /tmp2/b10902069/adl_hw1/multiple_choice_dir/special_tokens_map.json
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1358/1358 [10:04<00:00,  2.25it/s]