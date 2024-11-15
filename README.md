# garrulous

Attention-Based Infinite Shakespeare Generator with Read-Aloud (TTS)

## Interactive GPT

### Description

Infinite text generation model, trained on the Shakespeare dataset. The model is trained using the GPT architecture. The model is trained using only the PyTorch library.
Several techniques were used to train the model, starting of with the bigram model, then the multi-layer perceptron model and finally the Attention model.

### Usage

```shell
$ python3 interactive_gpt.py [-h] [--train] [--generate] [--model_path MODEL_PATH]
                              [--starter_text STARTER_TEXT] [--output_size OUTPUT_SIZE]
                              [--output_path OUTPUT_PATH] [--tts] [--voice_model VOICE_MODEL]
                              [--audio_path AUDIO_PATH]

> Train the model
$ python3 interactive_gpt.py --train --model_path <model_path>

> Generate text
$ python3 interactive_gpt.py --generate --model_path <model_path> --starter_text <starter_text> --output_size <output_size> --output_path <output_path>

> Infinite text generation
$ python3 interactive_gpt.py --generate --model_path <model_path> --starter_text <starter_text> --output_path <output_path>

> Generate text and synthesize to audio
$ python3 interactive_gpt.py --generate --tts --model_path <model_path> --starter_text <starter_text> --output_size <output_size> --output_path <output_path> --voice_model <voice_model> --audio_path <audio_path>
```

## Rant

### Description

A Python script to stream text and play corresponding audio in real-time. This script reads a text file and streams its content to the terminal character by character while simultaneously playing the corresponding audio file. The script uses multithreading to handle text and audio streams concurrently.

### Usage

```shell
$ python3 spit.py [-h] [--text_file TEXT_FILE] [--audio_file AUDIO_FILE]
                                   [--text_delay TEXT_DELAY]

> Stream text and play audio with default files
$ python3 spit.py

> Stream text and play audio with custom files
$ python3 spit.py --text_file <text_path> --audio_file <audio_path>

> Adjust text streaming speed
$ python3 spit.py --text_delay 0.07
```
