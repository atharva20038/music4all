import numpy as np
import audioldm_eval.metrics.distance as distance 
import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy
from multiprocessing.dummy import Pool as ThreadPool

SAMPLE_RATE = 16000


def load_audio_task(fname):
    try:
        wav_data, sr = sf.read(fname, dtype="int16")
    except Exception as e:
        print(e)
        wav_data = np.zeros(160000)
        sr = 16000
    assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if sr != SAMPLE_RATE:
        if SAMPLE_RATE == 16000 and sr == 32000:
            wav_data = wav_data[::2]
        else:
            wav_data = resampy.resample(wav_data, sr, SAMPLE_RATE)

    return wav_data, SAMPLE_RATE


class CMMD() : 
    def __init__(self, device, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8) -> None:
        self.device = device
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
        self.model = self.model.to(self.device)
        

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def get_embeddings(self, x, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        # x = x.to(self.device)
        embd_lst = []
        if isinstance(x, list):
            try:
                for audio, sr in tqdm(x, disable=(not self.verbose)):
                    print
                    embd = self.model.forward(audio, sr)
                    if self.model.device == torch.device("cuda"):
                        embd = embd.cpu()
                    embd = embd.detach().numpy()
                    embd_lst.append(embd)
            except Exception as e:
                print(
                    "[Frechet Audio Distance] get_embeddings throw an exception: {}".format(
                        str(e)
                    )
                )
        elif isinstance(x, str):
            if self.verbose:
                print("Calculating the embedding of the audio files inside %s" % x)
            try:
                for i, fname in tqdm(
                    enumerate(os.listdir(x)), disable=(not self.verbose)
                ):
                    if fname.endswith(".wav"):
                        if limit_num is not None and i > limit_num:
                            break

                        try:
                            audio, sr = load_audio_task(os.path.join(x, fname))
                            # audio_tensor = torch.from_numpy(audio).to(self.device)  # Ensure audio is a tensor and moved to the correct device
                            self.model = self.model.to(self.device)
                            
                            embd = self.model.forward(audio,self.device, sr)
                            if self.device == torch.device("cuda:1"):  # or specify the actual device, e.g., torch.device("cuda:0")
                                embd = embd.cpu()
                            embd = embd.detach().numpy()
                            embd_lst.append(embd)

                        except Exception as e:
                            print(e, fname)
                            continue
            except Exception as e:
                print(
                    "[Frechet Audio Distance] get_embeddings throw an exception: {}".format(
                        str(e)
                    )
                )
        else:
            raise AttributeError

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=True)
        return mu, sigma
    
    def score(self, background_dir, eval_dir, store_embds=False, limit_num=None):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        # audio_background = self.__load_audio_files(background_dir)
        # audio_eval = self.__load_audio_files(eval_dir)
        embds_background = self.get_embeddings(background_dir, limit_num=limit_num)
        embds_eval = self.get_embeddings(eval_dir, limit_num=limit_num)
        
        print(embds_background.shape)
        print(embds_eval.shape)
        
        print("Got embeddings")

        if store_embds:
            np.save("embds_background.npy", embds_background)
            np.save("embds_eval.npy", embds_eval)

        if len(embds_background) == 0:
            print(
                "[Frechet Audio Distance] background set dir is empty, exitting..."
            )
            return -1

        if len(embds_eval) == 0:
            print("[Frechet Audio Distance] eval set dir is empty, exitting...")
            return -1
        
        norm_1 = np.linalg.norm(embds_background, axis=1)
        norm_2 = np.linalg.norm(embds_eval, axis=1)
        
        print("With Norm")
        
        norm_1 = np.repeat(norm_1[:, np.newaxis], 2, axis=1)
        print(norm_1)
        
        embds_background /= norm_1
        embds_eval /= norm_2
    
        val = distance.mmd(embds_eval, embds_background)
        return val.numpy()
    
    def score_embed(self, embds_background, embds_eval, store_embds=False, limit_num=None):
        # background_dir: generated samples
        # eval_dir: groundtruth samples
        # try:
            # audio_background = self.__load_audio_files(background_dir)
            # audio_eval = self.__load_audio_files(eval_dir)
            # embds_background = self.get_embeddings(background_dir, limit_num=limit_num)
            # embds_eval = self.get_embeddings(eval_dir, limit_num=limit_num)
            
            # print(embds_background.shape)
            # print(embds_eval.shape)
            
            # print("Got embeddings")

            # if store_embds:
            #     np.save("embds_background.npy", embds_background)
            #     np.save("embds_eval.npy", embds_eval)

            # if len(embds_background) == 0:
            #     print(
            #         "[Frechet Audio Distance] background set dir is empty, exitting..."
            #     )
            #     return -1

            # if len(embds_eval) == 0:
            #     print("[Frechet Audio Distance] eval set dir is empty, exitting...")
            #     return -1

        
        print("Got embeddings")

        if store_embds:
            np.save("embds_background.npy", embds_background)
            np.save("embds_eval.npy", embds_eval)

        if len(embds_background) == 0:
            print(
                "[Frechet Audio Distance] background set dir is empty, exitting..."
            )
            return -1

        if len(embds_eval) == 0:
            print("[Frechet Audio Distance] eval set dir is empty, exitting...")
            return -1
        
        embds_eval = np.array(embds_eval)
        embds_background = np.array(embds_background)
        
        norm_1 = np.linalg.norm(embds_background, axis=1)
        norm_2 = np.linalg.norm(embds_eval, axis=1)
        
        # print("With Norm")
        
        norm_1 = np.repeat(norm_1[:, np.newaxis], embds_eval.shape[1], axis=1)
        norm_2 = np.repeat(norm_2[:, np.newaxis], embds_eval.shape[1], axis=1)
        
        embds_background /= norm_1
        embds_eval /= norm_2
        
        val = distance.mmd(embds_eval, embds_background)
        return val.numpy()

