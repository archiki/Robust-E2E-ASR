import argparse
import json
import os
import random
import time
import pdb
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from warpctc_pytorch import CTCLoss

from data.data_loader_noisy import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model_split_adversary import DeepSpeech, supported_rnns, NoiseClassifier
from test_noisy import evaluate
from utils import reduce_tensor, check_loss

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
					help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
					help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
					help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--continue-noise-from', default='')
parser.add_argument('--finetune', dest='finetune', action='store_true',
					help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
					help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=1-1/8, type=float, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0,
					help='Minimum noise level SNR to sample from. ', type=int)
parser.add_argument('--noise-max', default=25,
					help='Maximum noise levels SNR to sample from', type=int)
parser.add_argument('--noise-step', default = 5, help='step  size in SNR levels', type =int)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
					help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
					help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
					help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
					help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
					help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--load-noise-model',  action='store_true')
parser.add_argument('--mtl-lambda', default =0.7, type=float)
parser.add_argument('--noise-model-path', type=str)
parser.add_argument('--test-noise-dir',type=str)
parser.add_argument('--scale',type=float,default=10)
parser.add_argument('--scale-anneal',type=float,default=1.05)
parser.add_argument('--rnn-split',type=int,default = 2)
parser.add_argument('--binary-noisy',action='store_true', default=False)
parser.add_argument('--only-fc',action='store_true', default=False)
parser.add_argument('--noise-clubbed',action='store_true', default=False)
parser.add_argument('--lr-factor', type=float, default=1.0)
parser.add_argument('--recog-factor', type=float, default=1.0)
parser.add_argument('--noise-factor', type=float, default=1.0)
parser.add_argument('--test-noise-prob', type=float, default=0.5)
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
	return x.cpu().numpy()


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


if __name__ == '__main__':
	args = parser.parse_args()

	# Set seeds for determinism
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	device = torch.device("cuda" if args.cuda else "cpu")
	args.distributed = args.world_size > 1
	main_proc = True
	device = torch.device("cuda" if args.cuda else "cpu")
	if args.distributed:
		if args.gpu_rank:
			torch.cuda.set_device(int(args.gpu_rank))
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
		main_proc = args.rank == 0  # Only the first proc should save models
	save_folder = args.save_folder
	os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

	loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
		args.epochs)
	best_wer = None
	if main_proc and args.visdom:
		visdom_logger = VisdomLogger(args.id, args.epochs)
	if main_proc and args.tensorboard:
		tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

	avg_loss, start_epoch, start_iter, optim_state = 0, 0, 0, None
	#pdb.set_trace()
	if args.continue_from:  # Starting from previous model
		print("Loading checkpoint model %s" % args.continue_from)
		package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
		model = DeepSpeech.load_model_package(package)
		labels = model.labels
		#pdb.set_trace()
		audio_conf = model.audio_conf
		audio_conf['noise_dir'] = args.noise_dir
		audio_conf['noise_prob'] = args.noise_prob
		audio_conf['noise_levels'] = list(range(args.noise_min,args.noise_max + args.noise_step,args.noise_step)) 
	
	 
		if not args.finetune:  # Don't want to restart training
			optim_state = package['optim_dict']
		#	if isinstance(optim_state, list):
			start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
			start_iter = package.get('iteration', None)
			if start_iter is None:
				start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
				start_iter = 0
			else:
				start_iter += 1
#            pdb.set_trace()
			avg_loss = int(package.get('avg_loss', 0))
			try:
				loss_results[:start_epoch], cer_results[:start_epoch], wer_results[:start_epoch] = package['loss_results'], package['cer_results'], \
													 package['wer_results']
			except: 
				loss_results[:start_epoch], cer_results[:start_epoch], wer_results[:start_epoch] = package['loss_results'][:start_epoch], package['cer_results'][:start_epoch], \
                                                                                                         package['wer_results'][:start_epoch]

			#pdb.set_trace()
			best_wer = wer_results[start_epoch - 1]
			if main_proc and args.visdom:  # Add previous scores to visdom graph
				visdom_logger.load_previous_values(start_epoch, package)
			if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
				tensorboard_logger.load_previous_values(start_epoch, package)
		# MTL part
		if args.binary_noisy :
			nclasses = 2
		elif args.noise_clubbed :
			nclasses = 7
		else:
			nclasses = 8

		noise_model = NoiseClassifier(rnn_hidden_size=args.hidden_size,
						   nb_layers=1, #only FC
						   rnn_type=supported_rnns[args.rnn_type.lower()],
						   bidirectional=args.bidirectional,nclasses=nclasses)
		if args.only_fc :
			noise_model = NoiseClassifier(rnn_hidden_size=args.hidden_size,
						   nb_layers=0, #only FC
						   rnn_type=supported_rnns[args.rnn_type.lower()],
						   bidirectional=args.bidirectional,nclasses=nclasses)
		if args.continue_noise_from:
			print('Loaded Noise Model')
			noise_package = torch.load(args.continue_noise_from, map_location=lambda storage, loc: storage)
			noise_model = NoiseClassifier.load_model_package(noise_package)
		
	else:
		with open(args.labels_path) as label_file:
			labels = str(''.join(json.load(label_file)))

		audio_conf = dict(sample_rate=args.sample_rate,
						  window_size=args.window_size,
						  window_stride=args.window_stride,
						  window=args.window,
						  noise_dir=args.noise_dir,
						  noise_prob=args.noise_prob,
						  noise_levels=list(range(args.noise_min,args.noise_max + args.noise_step,args.noise_step)))

		rnn_type = args.rnn_type.lower()
		assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
		noise_model = NoiseClassifier(rnn_hidden_size=args.hidden_size,
						   nb_layers=args.hidden_layers,
						   rnn_type=supported_rnns[rnn_type],
						   bidirectional=args.bidirectional)

		model = DeepSpeech(rnn_hidden_size=args.hidden_size,
						   nb_layers=args.hidden_layers,
						   labels=labels,
						   rnn_type=supported_rnns[rnn_type],
						   audio_conf=audio_conf,
						   bidirectional=args.bidirectional)
	#ifNoisy = ['Noise', 'None']
	#snr_names = [str(x) for x in range(args.SNR_start, args.SNR_stop + args.SNR_step, args.SNR_step)]
	noises_names = ['Babble', 'AirportStation', 'Car', 'MetroSubway', 'CafeRestaurant', 'Traffic', 'ACVacuum', 'None']
	clubbed_dict = {'Babble':0, 'AirportStation':0, 'Car':2, 'MetroSubway':2, 'CafeRestaurant':0, 'Traffic':2, 'ACVacuum':4, 'None':6}
	snr_labels = {}
	for n in range(args.noise_min, args.noise_max + args.noise_step, args.noise_step):
		if(n <= 10):
			snr_labels[str(n)] = 0
		else:
			snr_labels[str(n)] = 1
	snr_labels['None'] = 0
	ifNoisy = {k:0 for k in noises_names[:-1]}
	ifNoisy['None'] = 1
	decoder = GreedyDecoder(labels)
	#test_audio_conf = audio_conf
	#test_audio_conf['noise_dir'] = args.test_noise_dir
	train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
									   normalize=True, augment=args.augment)
	test_audio_conf = audio_conf.copy()
	test_audio_conf ['noise_dir'] = args.test_noise_dir
	test_audio_conf ['noise_prob'] = args.test_noise_prob
	test_dataset = SpectrogramDataset(audio_conf=test_audio_conf, manifest_filepath=args.val_manifest, labels=labels,
									  normalize=True, augment=False)
	if not args.distributed:
		train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
	else:
		train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
													num_replicas=args.world_size, rank=args.rank)
	train_loader = AudioDataLoader(train_dataset,
								   num_workers=args.num_workers, batch_sampler=train_sampler)
	test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
								  num_workers=args.num_workers)

	if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
		print("Shuffling batches for the following epochs")
		train_sampler.shuffle(start_epoch)

	model = model.to(device)
	noise_model = noise_model.to(device)
	parameters = [{'params':model.parameters()},{'params':noise_model.parameters()}]
	
#    parameters = noise_model.parameters()

	#pdb.set_trace()
	# frozen_parameters = [{'params':model.rnns[4].parameters(), 'lr': 0.5*args.lr}, {'params':model.fc[0].parameters(), 'lr': 0.5*args.lr}, \
	# 					{'params':model.rnns[3].parameters(), 'lr': 0.5*args.lr}, {'params':model.rnns[2].parameters()}, \
	# 					{'params':model.rnns[1].parameters()}, {'params':model.rnns[0].parameters()}, {'params':model.conv.parameters()}]
	# #pdb.set_trace()
	# parameters = frozen_parameters
	# parameters.append({'params':noise_model.parameters()})

	feature_parameters = [{'params':model.rnns[2].parameters()},{'params':model.rnns[1].parameters()}, \
						  {'params':model.rnns[0].parameters()}, {'params':model.conv.parameters()}]
	# #pdb.set_trace()
	recog_parameters = [{'params':model.rnns[4].parameters()}, {'params':model.fc[0].parameters()}, \
						{'params':model.rnns[3].parameters()}]

	noise_parameters = noise_model.parameters()
	feature_optimizer = torch.optim.SGD(feature_parameters, lr=args.lr_factor*args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-5)
	recog_optimizer = torch.optim.SGD(recog_parameters, lr=args.recog_factor*args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-5)
	noise_optimizer = torch.optim.SGD(noise_parameters, lr=args.noise_factor*args.lr, momentum=args.momentum, nesterov=True, weight_decay=1e-5)
	
	if optim_state is not None:
		if isinstance(optim_state, list):

			feature_optimizer.load_state_dict(optim_state[0])
			recog_optimizer.load_state_dict(optim_state[1])
			noise_optimizer.load_state_dict(optim_state[2])

	[model, noise_model], [feature_optimizer, recog_optimizer, noise_optimizer] = amp.initialize([model, noise_model], \
									  [feature_optimizer, recog_optimizer, noise_optimizer],
									  opt_level=args.opt_level,
									  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
									  loss_scale=args.loss_scale)
	if args.distributed:
		model = DistributedDataParallel(model)
	print(model)
	print("Number of parameters: %d" % DeepSpeech.get_param_size(model))
#    for param in model.parameters(): param.requires_grad = False
	criterion = CTCLoss()
	criterion2 = torch.nn.CrossEntropyLoss()
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	noise_losses = AverageMeter()
	eval_accuracy_tracker = []
	best_epoch_tracker = []
	for epoch in range(start_epoch, args.epochs):
	   # model.audio_conf['noise_dir'] = args.noise_dir
		model.train()
		noise_model.train()
		end = time.time()
		start_epoch_time = time.time()
		for i, (data) in enumerate(train_loader, start=start_iter):
			#break
			if i == len(train_sampler):
				
				break
			inputs, targets, input_percentages, target_sizes, fileenames, accents, noises = data
			input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
			# measure data loading time
#            pdb.set_trace()
			data_time.update(time.time() - end)
			inputs = inputs.to(device)
			#pdb.set_trace()
			if_noisy_target = [ifNoisy[x] for x in noises]
			if_noisy_target = torch.FloatTensor(if_noisy_target).to(device).long()
			noise_target = [noises_names.index(x) for x in noises]
			if args.noise_clubbed:
				noise_target = [clubbed_dict[x] for x in noises]
				snr_target = [snr_labels[x] for x in accents]
				noise_target = np.array(noise_target) + np.array(snr_target)
			noise_target = torch.FloatTensor(noise_target).to(device)
			noise_target = noise_target.long()

			out,rnn_out, output_sizes = model(inputs, input_sizes,rnn_split = args.rnn_split)
			out = out.transpose(0, 1)  # TxNxH
			lengths = output_sizes.float().to(device)
			#if args.noise_clubbed: pdb.set_trace()
			
			#loss_noise = loss_noise / inputs.size(0)
			#loss = loss_noise
			float_out = out.float()
			loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
			loss = loss / inputs.size(0)  # average the loss by minibatch

#            pdb.set_trace()
			#loss = loss 
			if args.distributed:
				loss = loss.to(device)
				loss_value = reduce_tensor(loss, args.world_size).item()
			else:
				loss_value = loss.item()
				#noise_loss_value = loss_noise.item()
			
			# Check to ensure valid loss was calculated
			#loss = (1 - args.mtl_labda)*loss + args.mtl_lambda*loss_noise
			valid_loss, error = check_loss(loss, loss_value)
			if valid_loss:
				feature_optimizer.zero_grad()
				recog_optimizer.zero_grad()
				# compute gradient

				with amp.scale_loss(loss, [feature_optimizer, recog_optimizer]) as scaled_loss:
					scaled_loss.backward(retain_graph = True)
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
				#torch.nn.utils.clip_grad_norm_(noise_model.parameters(), args.max_norm)
				recog_optimizer.step()
				feature_optimizer.step()
			else:
				print(error)
				print('Skipping grad update')
				loss_value = 0
			del loss, out, float_out

			avg_loss += loss_value
			#avg_noise_loss += noise_loss_value
			losses.update(loss_value, inputs.size(0))
			noise_out = noise_model(rnn_out, lengths)
			del lengths
			float_noise = noise_out.float() 
			#float_out = out.float()  # ensure float32 for loss
			#loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
			#loss = loss / inputs.size(0)  # average the loss by minibatch
			if args.binary_noisy:
				loss_noise = criterion2(float_noise, if_noisy_target).to(device)
			else:
				loss_noise = criterion2(float_noise, noise_target).to(device)

			noise_optimizer.zero_grad()
			feature_optimizer.zero_grad()
				# compute gradient
			noise_loss_value = loss_noise.item()
			with amp.scale_loss(loss_noise, [feature_optimizer, noise_optimizer]) as scaled_loss:
				scaled_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
			torch.nn.utils.clip_grad_norm_(noise_model.parameters(), args.max_norm)
			noise_optimizer.step()
			feature_optimizer.step()

			noise_losses.update(noise_loss_value, inputs.size(0))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			optimizers = [feature_optimizer, recog_optimizer, noise_optimizer]
			if not args.silent:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Noise Loss {nloss.val:.4f} ({nloss.avg:.4f})\t'.format(
					(epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses, nloss=noise_losses))
			if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
				file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
				noise_file_path = '%s/deepspeech_noiseclassifier_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)

				print("Saving checkpoint model to %s" % file_path)
				torch.save(DeepSpeech.serialize(model, optimizer=optimizers, epoch=epoch, iteration=i,
												loss_results=loss_results,
												wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
						   file_path)
				torch.save(NoiseClassifier.seriaize(noise_model), noise_file_path)
			del loss_noise, noise_out, float_noise

		avg_loss /= len(train_sampler)

		epoch_time = time.time() - start_epoch_time
		print('Training Summary Epoch: [{0}]\t'
			  'Time taken (s): {epoch_time:.0f}\t'
			  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))
		#pdb.set_trace()
		start_iter = 0  # Reset start iteration for next epoch
		with torch.no_grad():
			#model.audio_conf['noise_dir'] = args.test_noise_dir
			wer, cer, output_data, eval_accuracy = evaluate(test_loader=test_loader,
											 device=device,
											 model=model,
											 decoder=decoder,
											 target_decoder=decoder,
											 rnn_split=args.rnn_split,ifNoiseClassifier=True,
											 ifNoiseBinary=args.binary_noisy,
											 noise_model = noise_model)
		loss_results[epoch] = avg_loss
		wer_results[epoch] = wer
		cer_results[epoch] = cer
		print('Validation Summary Epoch: [{0}]\t'
			  'Average WER {wer:.3f}\t'
			  'Average CER {cer:.3f}\t'
			  'Noise Classifier Accuracy {acc:.3f}\t'.format(
			epoch + 1, wer=wer, cer=cer, acc=eval_accuracy))
		eval_accuracy_tracker.append([epoch,eval_accuracy])
		if(epoch == args.epochs - 1):
			print(eval_accuracy_tracker)
			print(best_epoch_tracker)
		values = {
			'loss_results': loss_results,
			'cer_results': cer_results,
			'wer_results': wer_results
		}
		if args.visdom and main_proc:
			visdom_logger.update(epoch, values)
		if args.tensorboard and main_proc:
			tensorboard_logger.update(epoch, values, model.named_parameters())
			values = {
				'Avg Train Loss': avg_loss,
				'Avg WER': wer,
				'Avg CER': cer
			}

		if main_proc and args.checkpoint:
			file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
			noise_file_path = '%s/deepspeech_noiseclassifier%d.pth.tar' % (save_folder, epoch + 1)
			torch.save(DeepSpeech.serialize(model, optimizer=optimizers, epoch=epoch, loss_results=loss_results,
											wer_results=wer_results, cer_results=cer_results),
					   file_path)
			torch.save(NoiseClassifier.serialize(noise_model), noise_file_path)
		# anneal lr
		for g in feature_optimizer.param_groups:
			g['lr'] = g['lr'] / args.learning_anneal
		for g in recog_optimizer.param_groups:
                        g['lr'] = g['lr'] / args.learning_anneal
		for g in noise_optimizer.param_groups:
                        g['lr'] = g['lr'] / args.learning_anneal
		print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
		args.scale = args.scale / args.scale_anneal
		print('Scale annealed to: {s:.4f}'.format(s=args.scale))

		if main_proc and ( (best_wer is None or best_wer > wer) ):
			print("Found better validated model, saving to %s" % args.model_path)
			torch.save(DeepSpeech.serialize(model, optimizer=optimizers, epoch=epoch, loss_results=loss_results,
											wer_results=wer_results, cer_results=cer_results)
					   , args.model_path)
			torch.save(NoiseClassifier.serialize(noise_model), args.noise_model_path)
			best_wer = wer
			avg_loss = 0
			best_epoch_tracker.append(epoch)

		if not args.no_shuffle:
			print("Shuffling batches...")
			train_sampler.shuffle(epoch)
