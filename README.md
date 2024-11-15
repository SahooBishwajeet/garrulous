# garrulous

Attention-Based Infinite Shakespeare Generator with Read-Aloud (TTS)

## Interactive GPT

### Description

Infinite text generation model, trained on the Shakespeare dataset. The model is trained using the GPT architecture. The model is trained using only the PyTorch library.
Several techniques were used to train the model, starting of with the bigram model, then the multi-layer perceptron model and finally the Attention model.

### Usage

```shell
$ python3 interactive_gpt.py [-h] [--train] [--generate] [--spit] [--model_path MODEL_PATH]

> Train the model
$ python3 interactive_gpt.py --train --model_path <model_path>

> Generate text
$ python3 interactive_gpt.py --generate --model_path <model_path>

> Infinite text generation
$ python3 interactive_gpt.py --spit --model_path <model_path>
```
