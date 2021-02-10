import argparse
import pdb
import re
import numpy as np
import torch
from tqdm import tqdm
import pickle
from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils_orig import load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=10, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser.add_argument('--test-noise',default=None,help='Provide test noise samples')
parser.add_argument('--SNR-start', default=None,type=int, help = 'Provide starting SNR for noise injection')
parser.add_argument('--SNR-stop', default=None,type=int, help = 'Provide ending SNR for noise injection')
parser.add_argument('--SNR-step', default=5,type=int, help = 'Provide SNR steps between intermediate SNRs for noise injection')

#parser.add_argument('--SNR', default=None,type=int, help = 'Provide SNR for noise injection')
parser = add_decoder_args(parser)


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False, wer_dict ={}, accent_noise_dict = {},SNR=60):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    accent_noise_dict = {'us':{},'indian':{},'canada':{},'australia':{},'england':{},'scotland':{},'african':{}, 'libri':{}}
    noises_names = ['Babble', 'AirportStation', 'Car', 'MetroSubway', 'CafeRestaurant', 'Traffic', 'ACVacuum', 'None']
    noise_snr = { x:{str(i):[] for i in range(args.SNR_start,args.SNR_stop + args.SNR_step,args.SNR_step)} for x in noises_names}
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes, filenames = data
#        pdb.set_trace()
        noises = [re.sub('\d+',"",x.split('_')[1]) for x in filenames]
        snrs = [re.sub('[a-z][A-Z]+',"",x.split('_')[2]) for x in filenames]

#        pdb.set_trace()
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        rep_path = '../data/representations_timit/'
        out, output_sizes = model(inputs, input_sizes)
       # print(filenames)
        #pdb.set_trace()
        if save_output:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy()))

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            #print(transcript)
#            pdb.set_trace()
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            wer_dict[filenames[x]] = {'wer':100.0*wer_inst/len(reference.split()), 'decoded op':transcript}
            #if(noises[x] in accent_noise_dict[accents[x]].keys()):
           # try:
                #print(noises[x])
               # accent_noise_dict[accents[x]][noises[x]].append(100*wer_inst/len(reference.split()))
          #  except:
           #     print(noises[x],'created')
            #else:
                #accent_noise_dict[accents[x]][noises[x]] = [100*wer_inst/len(reference.split())]
            #pdb.set_trace()
           # pdb.set_trace()
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
#            pdb.set_trace()
            if snrs[x] != 'None':
                noise_snr[noises[x]][snrs[x]].append(100*float(wer_inst) / len(reference.split()))

            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
        #pdb.set_trace()
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
   # pdb.set_trace()
    #with open('/workspace/data/wers.pickle', 'wb+') as f:
        #pickle.dump(wer_dict, f)
    #with open('/workspace/data/accent-noise/libri/accent-noise-SNR-{}.pkl'.format(str(SNR)), 'wb+') as g:
        #pickle.dump(accent_noise_dict, g)
    if print_summary:
         print("NoiseType \t \t"+' \t'.join([str(x) for x in range(args.SNR_start, args.SNR_stop + args.SNR_step, args.SNR_step)]))
         for noise in noises_names[:-1]:
            track = []

            for snr in range(args.SNR_start, args.SNR_stop + args.SNR_step, args.SNR_step):
                track.append(np.mean(np.array(noise_snr[noise][str(snr)])))
                #print('{} \t {} \t {}'.format(noise, str(snr), np.mean(np.array(noise_snr[noise][str(snr)]))))
            print(noise+ ' \t \t' + ' \t'.join([str(np.round(x, decimals=1)) for x in track]))

    return wer * 100, cer * 100, output_data


if __name__ == '__main__':
    args = parser.parse_args()
    print_summary = True
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)
    wer_dict = {}
    accent_noise_dict = {}
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
#    pdb.set_trace()
    audio_conf_noise = model.audio_conf
    audio_conf_noise['noise_dir'] = args.test_noise
   # audio_conf_noise['noise_levels'] = args.SNR
    audio_conf_noise['noise_prob'] = 0
    test_dataset = SpectrogramDataset(audio_conf=audio_conf_noise, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    wer, cer, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half, wer_dict= wer_dict,SNR=args.SNR)

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    if args.save_output is not None:
        np.save(args.save_output, output_data)
