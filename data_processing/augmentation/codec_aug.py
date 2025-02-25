import io
import random
import ffmpeg
import numpy as np
import soundfile as sf
from scipy import signal
from pydub import AudioSegment

def convert_audio_on_the_fly_16K(audio_data: np.ndarray, samplerate: int, codec: str = "mp3", quality: int = 0):
    # Create a BytesIO buffer to store the FLAC audio data in memory
    with io.BytesIO() as input_buffer:
        # Save the given audio_data as FLAC format temporarily in the buffer
        sf.write(input_buffer, audio_data, samplerate, format='FLAC')
        input_buffer.seek(0)  # Move the pointer back to the beginning
        
        # Create another buffer to store the converted data
        output_buffer = io.BytesIO()

        # Set the appropriate FFmpeg options for different codecs
        if codec == "mp3":
            output_kwargs = {'format': 'mp3', 'q:a': quality}  # VBR for MP3
        elif codec == "ogg":
            output_kwargs = {'format': 'ogg', 'q:a': quality}  # VBR for Vorbis (OGG)
        elif codec == "aac":
            output_kwargs = {'format': 'adts', 'b:a': f'{quality}k'}  # CBR for AAC (quality is in kbps)
            #output_kwargs = {'format': 'adts', 'q:a': quality} 
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        # Use ffmpeg to handle the conversion and output the result to the buffer
        process = (
            ffmpeg
            .input('pipe:0', format='flac')  # Treat the stdin data as FLAC format
            .output('pipe:1', **output_kwargs)  # Apply codec-specific options
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        
        # Pass the input data to the process
        output, err = process.communicate(input=input_buffer.getvalue())
        
        # Check for ffmpeg errors
        if process.returncode != 0:
            print(f"FFmpeg error: {err.decode('utf-8')}")
            raise RuntimeError("FFmpeg failed to convert the audio data.")
        
        # Write the resulting converted data into the output buffer
        output_buffer.write(output)
        output_buffer.seek(0)  # Move the pointer back to the beginning
        
        # Read the data back using pydub's AudioSegment
        try:
            audio = AudioSegment.from_file(output_buffer, format=codec)
        except Exception as e:
            print(f"pydub error: {str(e)}")
            raise e
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Reshape if stereo or multi-channel
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
        
        return samples, output_buffer #audio.frame_rate, 

def alaw_compress(x):
    """Apply A-law compression."""
    A = 87.6  # A-law constant
    abs_x = np.abs(x)
    compressed = np.where(abs_x < (1 / A), A * abs_x / (1 + np.log(A)), (1 + np.log(A * abs_x)) / (1 + np.log(A)))
    compressed = np.sign(x) * compressed
    return compressed

def ulaw_compress(x, u=255):
    """Apply µ-law compression."""
    abs_x = np.abs(x)
    compressed = np.sign(x) * np.log(1 + u * abs_x) / np.log(1 + u)
    return compressed

def resample_audio(audio_data: np.ndarray, original_sr, target_sr):
    """Resample audio data to the target rate."""
    
    num_samples = round(len(audio_data) * float(target_sr) / original_sr)
    resampled_audio = signal.resample(audio_data, num_samples)
    return resampled_audio


def convert_audio_to_alaw(audio_data, sample_rate, target_sr=8000):
    """Convert audio to A-law format and downsample to 8kHz."""
    # Downsample from 16kHz to 8kHz
    audio_data_8k = resample_audio(audio_data, original_sr=sample_rate, target_sr=target_sr)
    
    # Normalize the audio data to [-1, 1] range if not already
    if np.max(np.abs(audio_data_8k)) > 1:
        audio_data_8k = audio_data_8k / np.max(np.abs(audio_data_8k))

    # Apply A-law compression
    compressed_audio = alaw_compress(audio_data_8k)
    
    return compressed_audio  # Return compressed data

def convert_audio_to_ulaw(audio_data, sample_rate, target_sr=8000):
    """Convert audio to µ-law format and downsample to 8kHz."""
    # Downsample from 16kHz to 8kHz
    audio_data_8k = resample_audio(audio_data, original_sr=sample_rate, target_sr=target_sr)
    
    # Normalize the audio data to [-1, 1] range if not already
    if np.max(np.abs(audio_data_8k)) > 1:
        audio_data_8k = audio_data_8k / np.max(np.abs(audio_data_8k))

    # Apply µ-law compression
    compressed_audio = ulaw_compress(audio_data_8k)
    
    return compressed_audio  # Return compressed data

class CodecAugmentationLight:
    """
    MP3    16kHz    50-190
    OGG    16kHz    64-192
    AAC    16kHz    32-128 (m4a)
    a-law    8kHz    8
    µ-law    8kHz    8  (u-law)
    """
    def __init__(self) -> None:
        self.codec_dict = {'mp3': [i for i in range(0, 8)], # 0: ~220-260 / 7: ~80-120 kbps
                            'ogg': [i for i in range(1, 9)], # 1: ∼80-96 / 8: ∼256-320 kbps
                            'aac': [32, 64, 96, 128], #[i for i in range(1, 6)], # 1: ∼20-32 / 5: ∼96-112 kbps
                            'a-law': [8],
                            'u-law': [8],} 
        self.codec_list = list(self.codec_dict.keys())
        
    def convert(self, audio_data, samplerate, codec=None, quality=None):
        codec, quality = self.selectCodec()
        buffer = None
        down_samplerate = 8000
        if codec == 'a-law':
            converted_audio = convert_audio_to_alaw(audio_data, samplerate, target_sr=down_samplerate)
            converted_audio = resample_audio(converted_audio, original_sr=down_samplerate, target_sr=samplerate)
        elif codec == 'u-law':
            converted_audio = convert_audio_to_ulaw(audio_data, samplerate, target_sr=down_samplerate)
            converted_audio = resample_audio(converted_audio, original_sr=down_samplerate, target_sr=samplerate)
        else:
            converted_audio, buffer = convert_audio_on_the_fly_16K(audio_data, samplerate, codec, quality)
        return self.normalize_audio(converted_audio)

    def selectCodec(self):
        #print(f"In selectCodec codec: {codec}, quality: {quality}")
        codec = random.choice(self.codec_list)
        quality = random.choice(self.codec_dict[codec])
        return codec, quality
 
    def normalize_audio(self, audio):
        """
        Normalize int16 or int32 audio data to the range of [-1.0, 1.0].
        
        :param audio_data: Input audio data (either int16, int32, or float64)
        :return: Normalized audio data in float64 format
        """
        if audio.dtype == np.int8: # int8 데이터를 float64로 정규화 (범위: -128 ~ 127)
            return audio.astype(np.float64) / 128.0
        elif audio.dtype == np.int16: # int16 데이터를 float64로 정규화 (범위: -32768 ~ 32767)
            return audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32: # int32 데이터를 float64로 정규화 (범위: -2147483648 ~ 2147483647)
            return audio.astype(np.float64) / 2147483648.0
        elif audio.dtype == np.float64: 
            return audio
        else:
            raise ValueError(f"Unsupported audio data type: {audio_data.dtype}")
    
    # # Only for the test
    # def test(self, audio_data, samplerate):
    #     for codec in self.codec_list:
    #         print(codec)
    #         for quality in self.codec_dict[codec]:
    #             converted_audio, sr, buffer = self.convert(audio_data, samplerate, codec, quality)
    #             save_audio(buffer, f'audio_{codec}_{quality}.{codec}')

class codecAugmentationFull:
    """
    MP3    16kHz    50-260
    OGG    16kHz    80-320
    AAC    16kHz    32-128 (m4a)
    a-law   8kHz    8
    µ-law   8kHz    8  (u-law)

    # FIXME (need to add more)
    OPUS   16kHz    6-30
            8kHz    4-20
    speex  16kHz    5.75-34.2
            8kHz    3.95-24.6
    amr    16kHz    6.6-23.05
            8kHz    4.75-12.20
    Encodec    16kHz    1.5-24.0
    """

    def __init__(self) -> None:
        self.codec_dict = {'mp3': [i for i in range(0,10)], # 0 ~ 9
                            'ogg': [i for i in range(1,10)], 
                            'aac': [i for i in range(32,129)], # 1, 2 
                            'a-law': [8],
                            'u-law': [8],} 
        self.codec_list = list(self.codec_dict.keys())
        

def save_audio(output_buffer, filename: str):
    """
    Save the converted audio from the BytesIO buffer to a file.
    
    :param output_buffer: BytesIO object containing the audio data.
    :param filename: The name of the file to save the output audio.
    """
    with open(filename, 'wb') as f:
        f.write(output_buffer.getvalue())
    print(f"Audio saved as {filename}")
    
# testing
if __name__ == '__main__':
    path = '/ssd/DB/ASVspoof5/B_1.flac'
    audio_data, sr = sf.read(path)
    codecAug = CodecAugmentationLight()
    codecAug.test(audio_data, sr)