import gradio as gr
from pytube import YouTube
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)


def transcribe(url):
    # get video and extract video
    def get_video(yt_url):
        try:
            video = YouTube(yt_url)
            video.streams.get_by_itag(22).download(filename='video.mp4')
            print('Video succesfully downloaded from Youtube')
        except Exception as e:
            print(f'Failed to download Youtube video \nerror : {e}')

    def audio_from_video(video_path):
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile('audio.wav')   
            video.close()
            audio.close()
        except Exception as e:
            print(f'Failed to extract audio from video \nerror : {e}')

    url = url
    video_path = './video.mp4'

    get_video(url)
    audio_from_video(video_path)


    # transcribe audio
   
    audio = 'audio.wav'

    text_audio = pipe(audio)

    chunks = text_audio['chunks']

    chunks_count = len(chunks)

    chunk_id = []
    timestamps = []
    texts = []
    start_time = []
    end_time = []


    for i in range(0, chunks_count):
        chunk_id.append(i)
        texts.append(chunks[i]['text'])
        start_time.append(chunks[i]['timestamp'][0])
        end_time.append(chunks[i]['timestamp'][1])

    chunk_length = []
    for i in range(0, chunks_count-1):
        chunk_length.append(round(end_time[i] - start_time[i], 3))

    output = list(zip(chunk_id, chunk_length, texts, start_time, end_time))

    sample_output_list = []
    for sublist in output:
        chunk_dict = {
            "chunk_id": sublist[0],
            "chunk_length": sublist[1],
            "text": sublist[2],
            "start_time": sublist[3],
            "end_time": sublist[4]
        }
        sample_output_list.append(chunk_dict)
    
    return sample_output_list

intf = gr.Interface(
    fn=transcribe,
    inputs = ["text"],
    outputs = ["text"]
)

intf.launch()