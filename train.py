import os
import copy
from tqdm import tqdm
from PIL import Image
import torch 
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from utils import all_gather
import torch.distributed as dist
import matplotlib.pyplot as plt


class ModelTrainer:
    args = None
    plm = None
    da = None
    loss = None
    classifier = None
    logger = None
    criterion = None
    optimizer = None
    lr_scheduler = None
    train_set = None
    sampler_bona = None
    train_loader_bona = None
    sampler_spoof = None
    train_loader_spoof = None
    val_loader = None
    eval_loader = None
    sampler_val = None
    num_val = None
    threshold = None
    lr_step = None
    best_validation_loss = 100

    def run(self):
        self.best_eer_19la = 100
        self.best_eer_21la = 100
        self.best_eer_21df = 100
        self.best_plm = copy.deepcopy(self.plm.state_dict())
        self.best_classifier = copy.deepcopy(self.classifier.state_dict())
        self.cnt_val = 0
        self.cnt_eval = 0
        self.end = False

        for epoch in range(1, self.args['epoch'] + 1):
            self.train(epoch)
            self._synchronize()
            self.validation(epoch)
            self._synchronize()
        
        self.evaluation(epoch, final=True, eval_only=False)
        

    def train(self, epoch):
        # set train mode
        if epoch <= self.args['T_0']:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False
            train_plm = False
        else:
            self.plm.train()
            for param in self.plm.parameters():
                param.requires_grad = True
            train_plm = True
        self.classifier.train()
        self.sampler_bona.set_epoch(epoch)
        self.sampler_spoof.set_epoch(epoch)

        # log lr
        if self.logger is not None:
            for p_group in self.optimizer.param_groups:
                lr = p_group['lr']
                self.logger.log_metric('lr', lr, step=epoch)
        
        # train
        count = 0
        loss_total = 0
        loader_bona = iter(self.train_loader_bona)
        loader_spoof = iter(self.train_loader_spoof)
        if len(self.train_loader_bona) < len(self.train_loader_spoof):
            big_loader = self.train_loader_spoof
            small_loader_iter = loader_bona
            small = 'bona'
        else:
            big_loader = self.train_loader_bona
            small_loader_iter = loader_spoof
            small = 'spoof'
        with tqdm(total=len(big_loader), ncols=90) as pbar:
            for i, (x_a, x_a_short, labels_a) in enumerate(big_loader):
                try:
                    x_b, x_b_short, labels_b = next(small_loader_iter)
                except StopIteration:
                    if small == 'bona': small_loader_iter = iter(self.train_loader_bona)
                    elif small == 'spoof': small_loader_iter = iter(self.train_loader_spoof)
                    x_b, x_b_short, labels_b = next(small_loader_iter)
                
                if labels_a[0] == 1: #if bonafide
                    x = torch.cat((x_a, x_b), dim=0)
                    x_short = torch.cat((x_a_short, x_b_short), dim=0)
                    labels = torch.cat((labels_a, labels_b), dim=0)
                    bona_size = x_a.size(0)
                else:
                    x = torch.cat((x_b, x_a), dim=0)
                    x_short = torch.cat((x_b_short, x_a_short), dim=0)
                    labels = torch.cat((labels_b, labels_a), dim=0)
                    bona_size = x_b.size(0)
                
                # to GPU
                x = x.to(device=self.args['device'], dtype=torch.float32)
                x_short = x_short.to(device=self.args['device'], dtype=torch.float32)
                labels = labels.to(self.args['device'])

                # data augmentation
                if self.da is not None:
                    x = self.da(x)
                    x_short = self.da(x_short)

                # feed forward
                if train_plm:
                    x_h = self.plm(x, output_hidden_states=True).hidden_states
                    x_hs = self.plm(x_short, output_hidden_states=True).hidden_states
                else:
                    with torch.set_grad_enabled(False):
                        x_h = self.plm(x, output_hidden_states=True).hidden_states
                        x_hs = self.plm(x_short, output_hidden_states=True).hidden_states
                
                ###################################### 
                self.optimizer.zero_grad()
                # forward classifier and get embeddings for EMA update
                loss1, _ , emb = self.classifier(torch.stack(x_h, dim=1), x_short=torch.stack(x_hs, dim=1), label=labels, bona_size=bona_size, return_embed=True)
                # optional proto-pull on bona-fide slice if enabled and using EMA loss
                proto_pull_w = self.args.get('proto_pull_w', 0.0)
                if proto_pull_w > 0 and hasattr(self.classifier.module.loss, 'proto_pull'):
                    emb_bona = emb[:bona_size, :]
                    loss_proto = self.classifier.module.loss.proto_pull(emb_bona, use_soft=(self.args.get('ema_assign','hard')=='soft'))
                    loss1 = loss1 + proto_pull_w * loss_proto
                loss1.backward()
                self.optimizer.step()
                
                # EMA prototype update using current bona-fide embeddings (no grad). Only rank 0 updates and broadcasts via DDP buffers implicitly.
                if hasattr(self.classifier.module.loss, 'update_prototypes') and bona_size > 0 and dist.get_rank() == 0:
                    with torch.no_grad():
                        self.classifier.module.loss.update_prototypes(emb[:bona_size, :].detach())
                ###################################### 
                

                if self.lr_step == 'iteration':
                    self.lr_scheduler.step()

                # log
                if self.args['flag_parent']:
                    count += 1
                    loss_total = loss_total + loss1.item()
                    if len(self.train_loader_bona) * 0.1 <= count:
                        self.logger.log_metric('Loss', loss_total / count)
                        count = 0
                        loss_total = 0
                
                    # pbar
                    desc = f'{self.args["name"]}-[{epoch}/{self.args["epoch"]}] | loss:{loss1.item():.3f} '
                    pbar.set_description(desc)
                    pbar.update(1)
                print(f"Epoch {epoch}, Iteration {i+1}/{len(big_loader)}, Loss: {loss1.item():.4f}")
            if self.lr_step == 'epoch':
                self.lr_scheduler.step()

    def validation(self, epoch):
        self.plm.eval()
        self.classifier.eval()
        self.sampler_val.set_epoch(epoch)

        # val
        count = 0
        loss_total = 0
        loss_total_all = 0  # Track full validation loss separately
        num_batches = 0
        with tqdm(total=len(self.val_loader), ncols=90) as pbar, torch.set_grad_enabled(False):
            for x, _, labels in self.val_loader:
                # to GPU
                x = x.to(torch.float32).to(self.args['device'], non_blocking=True)
                labels = labels.to(self.args['device'])

                # feed forward
                x = self.plm(x, output_hidden_states=True).hidden_states
                loss, _  = self.classifier(torch.stack(x, dim=1), label=labels)
                
                # Accumulate full validation loss
                loss_total_all += loss.item()
                num_batches += 1
                
                # log
                if self.args['flag_parent']:
                    count += 1
                    loss_total = loss_total + loss.item()
                    if len(self.val_loader) * 0.1 <= count:
                        self.logger.log_metric('Val_Loss', loss_total / count)
                        count = 0
                        loss_total = 0
                
                    # pbar
                    desc = f'{self.args["name"]}-[{epoch}/{self.args["epoch"]}] | val_loss:{loss.item():.3f} '
                    pbar.set_description(desc)
                    pbar.update(1)
                print(f"Epoch {epoch}, Validation Iteration {count}/{len(self.val_loader)}, Loss: {loss.item():.4f}")
        # Calculate average validation loss over all batches
        avg_val_loss = loss_total_all / num_batches
        print(f"Epoch {epoch}, Average Validation Loss: {avg_val_loss:.4f}")
        if self.best_validation_loss > avg_val_loss:
            self.best_validation_loss = avg_val_loss
            self.cnt_val = 0
            self.save_best_model(epoch, append=False, option=None)
        self._synchronize()

    def evaluation(self, epoch, final=False, avg=False, eval_only=False):
        if final:
            self.plm.load_state_dict(self.best_plm)
            self.classifier.load_state_dict(self.best_classifier)
        self.plm.eval()
        self.classifier.eval()
        

        eer_19la, _, _, _ = self.test('19LA', epoch)
        print(f"19LA EER: {eer_19la}")

        eer_21la, _, _, _ = self.test('21LA', epoch)
        print(f"21LA EER: {eer_21la}")
     
        # eer_21df, _, _, _ = self.test('21DF', epoch)
        # print(f"21DF EER: {eer_21df}")
        
        if not eval_only:
            self.cnt_eval += 1
            if eer_19la < self.best_eer_19la:
                self.best_eer_19la = eer_19la
                self.cnt_eval = 0
            if eer_21la < self.best_eer_21la:
                self.best_eer_21la = eer_21la
                self.cnt_eval = 0
            # if eer_21df < self.best_eer_21df:
            #     self.best_eer_21df = eer_21df
            #     self.cnt_eval = 0
            self.save_best_model(epoch, append=False, option='21la')
       

    def save_best_model(self, epoch=0, append=False, option='19pa'):  
        self.best_plm = copy.deepcopy(self.plm.state_dict())
        self.best_classifier = copy.deepcopy(self.classifier.state_dict())      
        if option == '19la':     
            if self.args['flag_parent']:
                self.logger.save_model(f'BestCLS_{epoch}_19LA', self.best_classifier)
        elif option == '21la':
            if self.args['flag_parent']:
                self.logger.save_model(f'BestCLS_{epoch}_21LA', self.best_classifier)
        # elif option == '21df':
        #     if self.args['flag_parent']:
        #         self.logger.save_model(f'BestCLS_{epoch}_21DF', self.best_classifier)
        

    def test(self, phase, epoch):
        # set test mode
        self.plm.eval()
        self.classifier.eval()

        # calculate score
        labels = {}
        scores = {}

        loader = self.eval_loader[phase]
        with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
            for x, attack_type, label in loader:
                # to GPU
                x = x.to(torch.float32).to(self.args['device'], non_blocking=True)
                
                # feed forward
                x = self.plm(x, output_hidden_states=True).hidden_states
                loss, score  = self.classifier(torch.stack(x, dim=1))
                
                # save score
                for i in range(score.size(0)):
                    try: 
                        labels[attack_type[i]]
                    except: 
                        labels[attack_type[i]] = []
                        scores[attack_type[i]] = []
                    
                    scores[attack_type[i]].append(score[i].item())
                    labels[attack_type[i]].append(label[i].item())
                pbar.update(1)
                
        self._synchronize()

        buffer1 = {}
        buffer2 = {}
        for key in labels.keys():
            buffer1[key] = all_gather(labels[key]) 
            buffer2[key] = all_gather(scores[key])
        labels = buffer1
        scores = buffer2
        self._synchronize()

        # calculate EER
        buffer1 = []
        buffer2 = []
        for k in labels.keys():
            buffer1 += scores[k]
            buffer2 += labels[k]
        

        eer = self.calculate_EER(buffer1, buffer2)
        precision, recall, f1_score = 0, 0, 0
        
        if self.args['flag_parent']:
            #exp_metrics = ['EER', 'Precision', 'Recall', 'F1_score']
            #results = [eer, precision, recall, f1_score]
            exp_metrics = ['EER']
            results = [eer]
            for i in range(len(exp_metrics)):
                self.logger.log_metric(f'{phase}/{exp_metrics[i]}', results[i], step=epoch)
            
        return eer, precision, recall, f1_score
    

    def calculate_EER(self, scores, labels):
        if len(scores) != len(labels):
            raise Exception('length between scores and labels is different')
        elif len(scores) == 0:
            raise Exception("There's no elements in scores")
            
        fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer * 100
    
    
    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()
