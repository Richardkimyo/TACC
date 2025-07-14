import time
import os
#from torch import nn
import torch.optim
from torch.utils import data
import argparse
import json
#import torchvision.transforms as transforms
from datasets.data.LEVIR_CC.LEVIRCC import LEVIRCCDataset

from models.Encoder import model
from models.Decoder import DecoderTransformer
from utils import *
from tqdm import tqdm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    """
    Testing.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint

    #snapshot_full_path = os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(args.savepath,map_location='cuda:0')
   
    
    encoder_trans = checkpoint['encoder_trans']
    decoder = checkpoint['decoder']
    
    
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        #LEVIR:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'test', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
  
    l_resize1 = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resize2 = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    # Epochs
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0
    total=0

    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        # Batches
        for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in enumerate(tqdm(test_loader)):

            # Move to GPU, if available
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            
            token_all = token_all.squeeze(0).cuda()
            #decode_lengths = max(token_all_len.squeeze(0)).item()
            # Forward prop.
           
            start_time = time.time()
            images1=1
            feat = encoder_trans(imgA,imgB) 
            seq = decoder.sample(feat,k=1)

            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                    img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            # # 判断有没有变化
          
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc+1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc+1
            end_time = time.time()
            total_time += (end_time - start_time)
            total_samples += images1
        test_time = time.time() - test_start_time
        avg_time_per_sample = total_time / total_samples
        fps = 1.0 / avg_time_per_sample        
        # captions
        # save_captions(args, word_map, hypotheses, references)
        print(f"Avg time per image pair: {avg_time_per_sample:.4f} seconds")
        print(f"Inference speed: {fps:.2f} image pairs per second")
        # Calculate evaluation scores
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))
 
        if len(nochange_references)>0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t' 
                'BLEU-4: {3:.4f}\t'  'Rouge: {4:.4f}\t' 'Cider: {5:.4f}\t''Meteor: {6:.4f}\t'
                .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4,Rouge, Cider,Meteor))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references)>0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t' 
                'BLEU-4: {3:.4f}\t'  'Rouge: {4:.4f}\t' 'Cider: {5:.4f}\t''Meteor: {6:.4f}\t'
                .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Cider,Meteor))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Testing:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t' 
            'BLEU-4: {4:.4f}\t'  'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t''Meteor: {7:.4f}\t'
            .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Rouge, Cider, Meteor))
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--data_folder', default='',help='folder with data files')
    parser.add_argument('--list_path', default='', help='path of the data lists')
    parser.add_argument('--token_folder', default='', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='path of the data lists')
    parser.add_argument('--data_name', default="",help='base name shared by data files.')
    parser.add_argument('--checkpoint', default='', help='path to checkpoint, None if none.')

    
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_dim',default=1024, help='the dimension of extracted features using different network:2048,512')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)

    parser.add_argument('--hidden_dim', type=int, default=512)
   
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Test
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="")
    
    args = parser.parse_args()
    main(args)