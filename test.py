import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import sys

from scipy import stats #correlation
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
            
        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
    corr_avg = 0 #Total correlation
    rms_avg = 0 #Total RMS Error
    all_val = 0 #Number of datapoints
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                              opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        #print(data['image'])
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = model.inference(data['label'], data['inst'], torch.empty(0)) #data['image'])
        

        data_im = data['image']


        data_im = data_im[~np.isnan(data_im)]


        data_gen = generated.data[0].cpu().float().numpy()[~np.isnan(generated.data[0].cpu().float().numpy())]


        try:
            corr, pval = stats.pearsonr(data_im, data_gen)
        except Exception as e: #If there is an error
            print("data_gen shape:", data_gen.shape)
            print("data_im shape:", data_im.shape)
            sys.stdout.flush()
            print(e)
            exit(1)
        #Update correlation and rmse
        corr_avg += corr
        rms_avg += np.sqrt(((data_gen - data_im.detach().cpu().numpy()) ** 2).mean() / (data_im.detach().cpu().numpy() **2).mean())
        all_val += 1





    print(corr_avg / all_val, opt.phase, "correlation")
    correlation = corr_avg/all_val #Calculate the correlation
    root_mean_square_error = rms_avg / all_val #Calculate the RMSE
    if opt.numpy_file != "None": #Store Correlation in numpy file opt.numpy_file
        if os.path.isfile(opt.numpy_file):
            file_open = np.load(opt.numpy_file)
            file_open[int(opt.which_epoch)] = correlation
            print("saving to numpy correlation", int(opt.which_epoch), file_open[int(opt.which_epoch)])
            np.save(opt.numpy_file, file_open)
        else: #If the file does not exist
            file_open = np.zeros((1000, 1))
            file_open[int(opt.which_epoch)] = correlation
            np.save(opt.numpy_file, file_open)
            print("making numpy", int(opt.which_epoch), file_open[int(opt.which_epoch)])
    
    if opt.numpy_file_rmse != "None": #Store RMSE in numpy file opt.numpy_file_rmse
        if os.path.isfile(opt.numpy_file_rmse):
            file_open = np.load(opt.numpy_file_rmse)
            file_open[int(opt.which_epoch)] = root_mean_square_error
            print("saving to numpy rmse", int(opt.which_epoch), file_open[int(opt.which_epoch)])
            np.save(opt.numpy_file_rmse, file_open)
        else: #If the file does not exist
            file_open = np.zeros((1000, 1))
            file_open[int(opt.which_epoch)] = root_mean_square_error
            np.save(opt.numpy_file_rmse, file_open)
            print("making numpy", int(opt.which_epoch), file_open[int(opt.which_epoch)])

#    webpage.save()
