import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import torch.nn as nn
import argparse
import json
from datetime import datetime
import pytz
#import torchvision.transforms as transforms
from datasets.data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
from models.Encoder import model
from models.Decoder import DecoderTransformer
from utils import *
from PIL import Image
import random
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from thop import profile




def main(args):
    """
    Training and validation.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.device_count()
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    best_bleu4 = 0.6 # BLEU-4 score right now
    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint
    if args.checkpoint is None:         
        encoder_trans = model()
        encoder_trans_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_trans.parameters()),
                                            lr=args.encoder_lr)
        decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
        
    else:
        checkpoint = torch.load(args.checkpoint)
        ##start_epoch = checkpoint['epoch'] + 1
        start_epoch = 0
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_trans = checkpoint['encoder_trans']
        encoder_trans_optimizer = checkpoint['encoder_trans_optimizer']
        
        
    # Move to GPU, if available
    encoder_trans = encoder_trans.cuda()
    
    decoder = decoder.cuda()
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=False)
    
    
   
    
    encoder_trans_lr_scheduler =torch.optim.lr_scheduler.MultiStepLR(encoder_trans_optimizer, milestones=[7,14,21,28,35,42,49,56,63], gamma=0.7)

    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[7,14,21,28,35,42,49,56,63], gamma=0.7)
    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs
    scaler = GradScaler()
    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_loader):
            #if id == 20:
            #    break
            start_time = time.time()
            decoder.train()  # train mode (dropout and batchnorm is used)
            encoder_trans.train()## 
            decoder_optimizer.zero_grad()
            encoder_trans_optimizer.zero_grad()
            

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            # Forward prop. 
            ##gpu_tracker.track()  
            with autocast():
                feat = encoder_trans(imgA,imgB) 
                
               
                scores, caps_sorted, decode_lengths, sort_ind = decoder(feat, token, token_len)
               
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                # Calculate loss
                
                loss = criterion(scores, targets)
                
            # Back prop.
            scaler.scale(loss).backward()
            # Clip gradients
            

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)
              

            
            scaler.step(decoder_optimizer)
           
            scaler.step(encoder_trans_optimizer)
            scaler.update()

            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            
            index_i += 1  
            if index_i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Loss: {4:.4f}\t'
                    'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs*len(train_loader),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
                                            np.mean(hist[index_i-args.print_freq:index_i-1,1]),
                                            np.mean(hist[index_i-args.print_freq:index_i-1,2])),
                    'decoder learning rate: {0:.6f}'.format(decoder_optimizer.param_groups[0]['lr']),
                    'encoder learning rate: {0:.6f}'.format(encoder_trans_optimizer.param_groups[0]['lr'])
                                            )
            #del scores, targets,loss
            #torch.cuda.empty_cache() 
              
        # One epoch's validation
        decoder.cuda()
        decoder.eval()  # eval mode (no dropout or batchnorm)
        encoder_trans.cuda()
        encoder_trans.eval()
        

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        
        with torch.no_grad():
            # Batches
            for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate((val_loader)):
                # Move to GPU, if available
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                feat= encoder_trans(imgA, imgB)
                seq = decoder.sample(feat, k=1)

               
                img_token = token_all.tolist()
               
                
                
                
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],img_token))  # remove <start> and pads
                
                
                



                references.append(img_tokens)
                
                #pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Calculate evaluation scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            beijing_tz = pytz.timezone('Asia/Shanghai')
            current_time1 = datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')   
            print('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t' 
                'BLEU-4: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t''Meteor:{7}\t' 'Time:{8}\t'
                .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4,  Rouge, Cider,Meteor,current_time1))
        
        #Adjust learning rate
        decoder_lr_scheduler.step()
        #print(decoder_optimizer.param_groups[0]['lr'])
        encoder_trans_lr_scheduler.step()   
        
            #print(encoder_optimizer.param_groups[0]['lr'])
        # Check if there was an improvement        
        if Bleu_4 > best_bleu4:
            #best_bleu4 = max(Bleu_4, best_bleu4)
            #save_checkpoint                
            print('Save Model,epoch: [{0}]\tBLEU-4: {1:.4f}\t'.format(epoch, Bleu_4))
            state = {
                    'encoder_trans': encoder_trans,  
                    'encoder_trans_optimizer':encoder_trans_optimizer, 
                    'decoder': decoder,
                    'decoder_optimizer': decoder_optimizer,
                    'epoch': epoch,
                    'bleu-4': Bleu_4
                    }
            # Get current time in Beijing
            beijing_tz = pytz.timezone('Asia/Shanghai')
            current_time = datetime.now(beijing_tz).strftime('%Y-%m-%d_%H-%M-%S')        
            model_name = str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+'Bleu_4_'+str(round(10000*best_bleu4))+'_'+current_time+'_'+str(epoch)+'.pth'
            torch.save(state, os.path.join(args.savepath, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')
    # Data parameters
    parser.add_argument('--data_folder', default='', help='folder with data files')
    parser.add_argument('--list_path', default='', help='path of the data lists')
    parser.add_argument('--token_folder', default='', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')
    
    
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    #parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=50, help='print training/validation stats every __ batches. ')
    # Training parameters

    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=32, help='batch_size for training')
    
    parser.add_argument('--encoder_dim',default=1024, help=' encoder feature ')
    
    parser.add_argument('--num_epochs', type=int, default=70, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=16e-5, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=16e-5, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="")
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
   
    parser.add_argument('--feature_dim', type=int, default=512)
    args = parser.parse_args()
    main(args)
   