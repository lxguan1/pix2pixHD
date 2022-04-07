import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import psutil
from guppy import hpy; h=hpy()
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


#from analysis.corr import corr #DANIEL ADDED THIS

def output_correlation(visuals):
    have_synth = False
    have_real = False 
    curr_synth = np.array([])
    curr_real = np.array([])
    #print("len(visuals.items()):", len(visuals.items()))
    for label, image_numpy in visuals.items():
        #print(f"label is + {label}")
        #curr_synth = np.array([])
        #curr_real = np.array([])
        #have_synth = False
        #have_real = False
        if label[0] == 's': #this is a synthesized image
            curr_synth = image_numpy
            #print(f"This is the image_numpy shape: {image_numpy.shape} " )
            have_synth = True
        elif label[0] == 'r': #this is a real image
            curr_real = image_numpy
            #print(f"This is the image_numpy shape: {image_numpy.shape}" )
            have_real = True

        if have_real and have_synth:
            have_real = False
            have_synth = False
            print (f"The correlation is : {corr(curr_real , curr_synth)}")
            


    return










def main(cont_train):
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    opt.continue_train =  cont_train
    print("opt:", opt.continue_train, flush = True)
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    #print(dataset)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size, flush=True)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    print("Visualizer created")
    if opt.fp16:    
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

       
    print("model created", dataset_size)
    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
 #   print("Start epoch")
    
    
    #for data in dataset:
       # print("data el:", data)

    for epoch in range(start_epoch, start_epoch + 1): #The epoch number inputted #opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
#        print("start dataset enumeration")
        start_loaddata = time.time()
        for i, data in enumerate(dataset, start=epoch_iter):
            if i == 0:
                print("loaded:", time.time() - start_loaddata, flush=True)
            #print("inside epoch enum data", flush=True)
  #          print("RAM objects:",h.heap(),flush=True)
            #print(h.heap().byid[0].sp, flush=True)
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

   #         print("img shape1:", data['image'].shape)
    #        print("inst shape1:", data['inst'].shape)
     #       print("feat shape1:", data['feat'].shape)

            #print("Utilization outside forward", flush=True)
            #print("CPU percentage:", psutil.cpu_percent())
            #print("RAM utilization:", psutil.virtual_memory().percent)
            #print("RAM value:", psutil.virtual_memory())

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), infer=save_fake)

            #print("img shape2:", data['image'].shape)
            #print("inst shape2:", data['inst'].shape)
            #print("feat shape2:", data['feat'].shape)
            #print("image:", np.unique(data['image']))
            #print("unique values:", len(np.unique(data['image'])))

            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            #print("Utilization inside forward")
            #print("CPU percentage:", psutil.cpu_percent())
            #print("RAM utilization:", psutil.virtual_memory().percent)
            #print("RAM value:", psutil.virtual_memory())

 
           # print(torch.cuda.memory_summary()) #Memory allocated
            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
            else:
                loss_G.backward()          
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step()        

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
 #               output_correlation(visuals) #DANIEL ADDED THIS
            ### save latest model
            if False: #total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')            
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
       
        # end of epoch 
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if (opt.gan_loss_numpy != None and opt.disc_loss_numpy != None):
            file_all_gan_loss = np.load(opt.gan_loss_numpy)
            file_all_disc_loss = np.load(opt.disc_loss_numpy)
            file_all_gan_loss[start_epoch] = loss_dict["G_GAN"].item()
            file_all_disc_loss[start_epoch] = loss_D.item()
            np.save(opt.gan_loss_numpy, file_all_gan_loss)
            np.save(opt.disc_loss_numpy, file_all_disc_loss)
            print("Saving Gan Loss", loss_dict["G_GAN"].item(), loss_D.item())

        ### save model for this epoch
        if True: #epoch % opt.save_epoch_freq == 0: 
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    if opt.which_epoch == 1: #Make a new model if epoch == 1
        main(False)
    else:
        main(True)
    #print(opt.continue_train)
    #for i in range(200):
    #    torch.cuda.empty_cache()
    #    if (i == 0):
    #        #opt.continue_train = False
    #        #print(opt.continue_train)
    #        main(False)
    #    else:
    #        #opt.continue_train = True
    #        main(True)
    #    print(i, "=========================================================================")
    #    #main()
    #    print("RAM objects:",h.heap(),flush=True)
