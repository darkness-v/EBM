import os
from dataclasses import dataclass

@dataclass
class Trial:
    path: str
    speaker: str
    type: str
    label: int

class ASVspoof2019_LA_Eval:
    def __init__(self, path):
        self.trials = []
        self.class_num = [0, 0]
        # trials
        for line in open(os.path.join(path, 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')).readlines():
            strI = line.split(' ')
            label = 1 if strI[4].replace('\n', '') == 'bonafide' else 0
            if strI[3] == '-':
                strI[3] = 'bonafide'
            item = Trial(
                path=os.path.join(path, 'LA/ASVspoof2019_LA_eval/flac', f'{strI[1]}.flac'), 
                speaker=strI[0], 
                type=strI[3],
                label=label
            )
            self.class_num[label] += 1
            self.trials.append(item)
        print(f'19LA eval #bonafide & #spoofed: {self.class_num[1]} / {self.class_num[0]}')

        #assert len(self.trials) == 134730