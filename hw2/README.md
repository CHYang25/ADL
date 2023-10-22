# ADL HW2 Fine-tuning mT5 - New Title Generation
Text-to-Text Transfer Transformer” (T5)

some notes:
- We only train the decoder, encoder is already finished
- mt5 is a multi-lingual T5
- what i have to do with my code:
    - import all packages and initialize all the tools
    - parse arguments
    - prepare and tokenize the data
    - generate the encoded data
    - train with the encoded data
    - validate and store the result
- After everything is done, continue to Reinforcement Learning.

development diary:
- 2023/10/22: finished parse_arg.py, tw_rouge package download, train_decoder.py import packages, and train run script files PYTHONPATH specification
- 

useful links:
- homework info: https://docs.google.com/presentation/d/1yJEQUtzFREeuEnkBTXei4SFEftnmnP3i05m0J5aGsg8/edit#slide=id.gd5d1d9d2d2_0_201
- Acclerator:
    - https://pypi.org/project/accelerate/
    - https://huggingface.co/docs/accelerate/basic_tutorials/migration
- mt5:
    - https://huggingface.co/google/mt5-small
    - https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    - https://huggingface.co/docs/transformers/main_classes/model#generation
    - mt5 papaer: https://arxiv.org/abs/2010.11934
- fine-tuning mt5:
    - https://github.com/KrishnanJothi/MT5_Language_identification_NLP/blob/main/MT5_fine-tuning.ipynb
    - metric: twrouge: https://github.com/moooooser999/ADL23-HW2