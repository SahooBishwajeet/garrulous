import os
import wave
import pyaudio
import time
import argparse
from threading import Thread


def play_audio_stream(audio_file_path):
    """Stream audio playback in real-time."""
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' does not exist.")
        return

    with wave.open(audio_file_path, "rb") as wav_file:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=audio.get_format_from_width(wav_file.getsampwidth()),
            channels=wav_file.getnchannels(),
            rate=wav_file.getframerate(),
            output=True,
        )

        print("\nPlaying audio stream...\n")
        chunk_size = 1024
        data = wav_file.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wav_file.readframes(chunk_size)

        stream.stop_stream()
        stream.close()
        audio.terminate()
    print("\nAudio playback finished.")


def stream_text(text_file_path, delay=0.07):
    """Stream text to the terminal character by character."""
    if not os.path.exists(text_file_path):
        print(f"Error: Text file '{text_file_path}' does not exist.")
        return

    print("Streaming text:\n")
    with open(text_file_path, "r") as file:
        text = file.read()

    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)  # Simulate streaming speed


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(
        description="Stream text and play corresponding audio in real-time"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="./output/generated_text.txt",
        help="Path to the text file (default: ./output/generated_text.txt)",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="./output/generated_audio.wav",
        help="Path to the audio file (default: ./output/generated_audio.wav)",
    )
    parser.add_argument(
        "--text_delay",
        type=float,
        default=0.07,
        help="Delay (in seconds) between characters when streaming text (default: 0.07)",
    )
    args = parser.parse_args()

    # Validate file paths
    text_file = args.text_file
    audio_file = args.audio_file
    text_delay = args.text_delay

    # Create threads for text streaming and audio playback
    audio_thread = Thread(target=play_audio_stream, args=(audio_file,))
    text_thread = Thread(target=stream_text, args=(text_file, text_delay))

    # Start threads
    audio_thread.start()
    text_thread.start()

    # Wait for both threads to finish
    audio_thread.join()
    text_thread.join()

    print("\nStreaming complete.")
