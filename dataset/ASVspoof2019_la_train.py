import os
from dataclasses import dataclass

@dataclass
class Item:
    path: str
    speaker: str
    label: int

@dataclass # init, repr, eq
class Trial:
    path: str
    speaker: str
    type: str
    label: int

class ASVspoof2019_LA_Train:
    def __init__(self, path):
        self.train_set = []
        self.dev_set = []
        self.class_weight = [0, 0]
        self.class_num = [0, 0]
        self.class_num_val = [0, 0]

        # train_set
        for line in open(os.path.join(path, 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')).readlines():
            strI = line.split(' ')
            label = 1 if strI[4].replace('\n', '') == 'bonafide' else 0
            if label == 0:
                item = Item(
                    path=os.path.join(path, f'LA/ASVspoof2019_LA_train/vocoder/hifi_gan/flac', f'{strI[1]}.flac'), 
                    speaker=strI[0], 
                    label = 0)
                self.class_weight[0] += 1
                self.class_num[0] += 1
                self.train_set.append(item)

            item = Item(
                path=os.path.join(path, f'LA/ASVspoof2019_LA_train/flac', f'{strI[1]}.flac'), 
                speaker=strI[0], 
                label=label)
            self.class_weight[label] += 1
            self.class_num[label] += 1
            self.train_set.append(item)
            

        self.class_weight[0] = len(self.train_set) / self.class_weight[0]
        self.class_weight[1] = len(self.train_set) / self.class_weight[1]  
        print(f'19LA+Codec train #bonafide & #spoofed: {self.class_num[1]} / {self.class_num[0]}')


        # dev_set
        for line in open(os.path.join(path, 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')).readlines():
            strI = line.split(' ')
            label = 1 if strI[4].replace('\n', '') == 'bonafide' else 0
            if strI[3] == '-':
                strI[3] = 'bonafide'
            item = Trial(
                path=os.path.join(path, 'LA/ASVspoof2019_LA_dev/flac', f'{strI[1]}.flac'), 
                speaker=strI[0], 
                type=strI[3],
                label=label
            )
            self.class_num_val[label] += 1
            self.dev_set.append(item)
        print(f'19LA dev #bonafide & #spoofed: {self.class_num_val[1]} / {self.class_num_val[0]}')

        #assert len(self.train_set) == 25380
        #assert len(self.dev_set) == 24844