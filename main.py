from utils.sound_sentiment import *
from utils.neural_translator import NeuralSentimentTranslator
from utils.diffusion_utils import *
import matplotlib.pyplot as plt
from os.path import join
import argparse
from pathlib import Path

if __name__ == "__main__":
    output_path = join(".", "outputs")

    parser = argparse.ArgumentParser(description="Music to Video Generation")

    parser.add_argument("--song", type=Path,
                        help="Song to use to generate", default=join(".", "data", "test1.wav"))
    parser.add_argument("--video", type=Path,
                        help="Output Video Path", default=join(output_path, "output.avi"))
    parser.add_argument("--frames", type=int,
                        help="Number of frames between", default=10)
    parser.add_argument("--window_size", type=int,
                        help="Window size of sampling", default=10)
    parser.add_argument("--debug", type=bool,
                        help="Show Debug Info", default=True)
    args = parser.parse_args()

    # 2 * window_size seconds, take mean of frames between samples => 2 * frames/window_size between different image types
    fps = int(2 * args.frames/args.window_size)

    sound_sentiment = SoundSentimentExtractor()
    attribute, x = sound_sentiment.extract_sentiment(
        args.song, args.window_size)
    neural_translator = NeuralSentimentTranslator()
    class_vectors, song_words = neural_translator.translate_sentiment(
        attribute)
    if args.debug:
        print(len(song_words))
        print(song_words)
        print(list(enumerate(song_words)))
        print(len(attribute))
        plt.plot(attribute[:, 0], label="valence")
        plt.plot(attribute[:, 1], label="arousal")
        plt.legend()
        plt.show()

    vae, unet, scheduler, tokenizer, text_encoder = get_vae(
    ), get_unet(), get_scheduler(), get_tokenizer(), get_text_encoder()

    images = generate_video(song_words, )
    grid = image_grid(imgs=images, rows=len(images), col=args.frames)
    grid.save(join(output_path, "grid.png"))
