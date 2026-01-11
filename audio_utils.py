from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_audio="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        output_audio,
        fps=16000,
        nbytes=2,
        codec="pcm_s16le",
        verbose=False,
        logger=None
    )
    return output_audio
