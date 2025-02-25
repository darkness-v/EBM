import os
from dataclasses import dataclass

@dataclass
class Trial:
    path: str
    type: str
    label: int

class ASVspoof2021_LA_Eval:
    def __init__(self, path):
        self.trials = []
        self.class_num = [0, 0]
        # trials
        # /data/ASVspoof2021_LA_eval/
        # /home/shin/exps/DB/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt
        # print(os.path.join(path, 'keys/LA/CM/trial_metadata.txt'))

        for i, line in enumerate(open(os.path.join(path, 'keys/LA/CM/trial_metadata.txt')).readlines()):
            strI = line.rstrip('\n').split(' ')
            if strI[7] != 'eval':
                continue
            label=1 if strI[4] == 'bonafide' else 0
            item = Trial(
                path=os.path.join(path, 'flac', f'{strI[1]}.flac'), 
                type=strI[4], 
                label=label
            )
            self.trials.append(item)
            self.class_num[label] += 1
        print(f'21LA eval #bonafide & #spoofed: {self.class_num[1]} / {self.class_num[0]}')

        #assert len(self.trials) == 148176
        
        