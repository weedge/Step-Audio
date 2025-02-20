import torchaudio
import argparse
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from utils import load_audio
import os


def main():
    parser = argparse.ArgumentParser(description="StepAudio Stream Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Base path for model files")
    parser.add_argument(
        "--synthesis-type", type=str, default="tts", help="Use tts or Clone for Synthesis"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for synthesis audios"
    )
    parser.add_argument(
        "--stream", type=str, default="static_batch", help="synthesis audios with streaming"
    )
    args = parser.parse_args()
    os.makedirs(f"{args.output_path}", exist_ok=True)

    encoder = StepAudioTokenizer(f"{args.model_path}/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(f"{args.model_path}/Step-Audio-TTS-3B", encoder)

    if args.synthesis_type == "tts":
        text = "（RAP）君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        output_audio, sr = tts_engine.static_batch_stream(text, "Tingting")
        torchaudio.save(f"{args.output_path}/output_tts.wav", output_audio, sr)
    else:
        clone_speaker = {
            "speaker": "test",
            "prompt_text": "叫做秋风起蟹脚痒，啊，什么意思呢？就是说这秋风一起啊，螃蟹就该上市了。",
            "wav_path": "examples/prompt_wav_yuqian.wav",
        }
        text_clone = "万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        output_audio, sr = tts_engine.static_batch_stream(text_clone, "", clone_speaker)
        torchaudio.save(f"{args.output_path}/output_clone.wav", output_audio, sr)


if __name__ == "__main__":
    main()
