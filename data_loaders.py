import numpy as np
import soundfile as sf
import random


from data_processing.augmentation.raw_boost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

from data_processing import CodecAugmentationLight
from torch.utils.data import Dataset, DataLoader, DistributedSampler

def separate_bona_spoof(train_db):
    bona_db = []
    spoof_db = []
    for item in train_db:
        if item.label == 1:
            bona_db.append(item)
        else: 
            spoof_db.append(item)
    return bona_db, spoof_db

# Separate bonafide and spoofed
def get_loaders(args, train_db, val_db, eval_db):
    bona_db, spoof_db = separate_bona_spoof(train_db)

    # get bonafide
    trains_set_bona = TrainSet(args, bona_db)
    train_sampler_bona = DistributedSampler(trains_set_bona, shuffle=True)
    train_loader_bona = DataLoader( 
        trains_set_bona,
        num_workers=args['num_workers'],
        batch_size=int(args['batch_size'] * args['ratio_bona']),
        pin_memory=True,
        sampler=train_sampler_bona,
    )
    
    # get spoofed
    trains_set_spoof = TrainSet(args, spoof_db)
    train_sampler_spoof = DistributedSampler(trains_set_spoof, shuffle=True)
    train_loader_spoof = DataLoader( 
        trains_set_spoof,
        num_workers=args['num_workers'],
        batch_size=int(args['batch_size'] * args['ratio_spoof']),
        pin_memory=True,
        sampler=train_sampler_spoof,
    )

    val_set = EnrollmentSet(args, val_db)
    val_sampler = DistributedSampler(val_set, shuffle=False)
    val_loader = DataLoader(
        val_set,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'] // args['num_seg'],
        sampler=val_sampler,
        pin_memory=False,
    )

    eval_loader = {}

    eval_set_19la = EnrollmentSet(args, eval_db['19LA'].trials)
    eval_sampler = DistributedSampler(eval_set_19la, shuffle=False)
    eval_loader['19LA'] = DataLoader(
        eval_set_19la,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'] // args['num_seg'],
        sampler=eval_sampler,
        pin_memory=False,
    )

    eval_set_21la = EnrollmentSet(args, eval_db['21LA'].trials)
    eval_sampler = DistributedSampler(eval_set_21la, shuffle=False)
    eval_loader['21LA'] = DataLoader(
        eval_set_21la,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'] // args['num_seg'],
        sampler=eval_sampler,
        pin_memory=False,
    )

    eval_set_21df = EnrollmentSet(args, eval_db['21DF'].trials)
    eval_sampler = DistributedSampler(eval_set_21df, shuffle=False)
    eval_loader['21DF'] = DataLoader(
        eval_set_21df,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'] // args['num_seg'],
        sampler=eval_sampler,
        pin_memory=False,
    )

    return train_loader_bona, train_sampler_bona, train_loader_spoof, train_sampler_spoof, \
           val_loader, eval_loader

class TrainSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.crop_size = args['num_train_frames'] * 160 - 40
        self.crop_size_short = args['num_train_frames_short'] * 160 - 40
        self.codec_aug = CodecAugmentationLight()
        self.p_codec_aug = 0.8
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        audio, sr = sf.read(item.path)

        if random.random() < self.p_codec_aug:
            codec_crop_size = int(self.crop_size * 1.2)
            if codec_crop_size < audio.shape[0]:
                audio = audio[:codec_crop_size]
            audio = self.codec_aug.convert(audio, sr)

        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        else:
            audio = audio[:self.crop_size]


        augtype = random.randint(0,8)
        if augtype == 1:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
        elif augtype == 2:
            audio = ISD_additive_noise(audio) # Impulsive noise
        elif augtype == 3:
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 4:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = ISD_additive_noise(audio) # Impulsive noise
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 5:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = ISD_additive_noise(audio) # Impulsive noise
        elif augtype == 6:
            audio = LnL_convolutive_noise(audio) # Convolutive noise 
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 7:
            audio = ISD_additive_noise(audio) # Impulsive noise
            audio = SSI_additive_noise(audio) # coloured additive noise
        elif augtype == 8:
            audio1 = LnL_convolutive_noise(audio) # Convolutive noise 
            audio2 = ISD_additive_noise(audio) # Impulsive noise
            audio = audio1 + audio2
            audio=normWav(audio, 0)


        # read short wav
        audio_short, _ = sf.read(item.path)

        if random.random() < self.p_codec_aug:
            codec_crop_size_short = int(self.crop_size_short * 1.2)
            if codec_crop_size_short < audio_short.shape[0]:
                audio_short = audio_short[:codec_crop_size_short]
            audio_short = self.codec_aug.convert(audio_short, sr)

        if audio_short.shape[0] < self.crop_size_short:
            shortage = self.crop_size_short - audio_short.shape[0]
            audio_short = np.pad(audio_short, (0, shortage), 'wrap')
        else:
            start = random.randint(0, audio_short.shape[0] - self.crop_size_short) 
            audio_short = audio_short[start: start + self.crop_size_short]


        augtype = random.randint(0,8)
        if augtype == 1:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
        elif augtype == 2:
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
        elif augtype == 3:
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 4:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 5:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
        elif augtype == 6:
            audio_short = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 7:
            audio_short = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = SSI_additive_noise(audio_short) # coloured additive noise
        elif augtype == 8:
            audio1 = LnL_convolutive_noise(audio_short) # Convolutive noise 
            audio2 = ISD_additive_noise(audio_short) # Impulsive noise
            audio_short = audio1 + audio2
            audio_short=normWav(audio_short, 0)


        return audio, audio_short, item.label


class EnrollmentSet(Dataset):
    def __init__(self, args, items):
        self.items = items    
        self.crop_size = args['num_test_frames'] * 160 - 40
        self.num_seg = args['num_seg']
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)
        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        else:
            audio = audio[:self.crop_size]

        return audio, item.type, item.label
    