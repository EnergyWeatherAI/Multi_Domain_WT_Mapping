import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from utils.loadsave import save_checkpoint
from utils.mapping_earlystop import mapped_target_avg_score, MappingEarlyStopper

import numpy.random
from data import data_corrupter

import json

class Trainer():
    '''
    This class is used to perform the actual training loops to train a domain mapping network. 
    train() can be called to fully train the network with provided models and dataloaders, see train_mapping.py 
    '''

    def __init__(self, config):
        '''
        Initializes a trainer instance with provided training settings. See train_mapping.py for an example.

        Args:
            config (dict): A dictionary containing general training settings, with the following key/value expectations:
                device (torch device): train the network on a specified cuda device or cpu
                save_dir (str): general save directory (folder) for saving generator checkpoints 
                n_wts (int): how many wind turbines are involved in the stargan/setup
                lambdas (dict): 
                    Hyper-parameters; lambda weights for the loss components:
                        cyc: cycle_loss weight, zero: zero_loss weight;
                        max: rated power loss weight, clsD: discriminator classifier loss weight;
                        clsG: generator class belonging weight
                max_powers (list): for each wt, the maximum power as a *normalized* value
        '''

        self.device = config["device"]
        self.save_dir = config["save_dir"]
        self.lambdas = config["lambdas"]
        self.max_powers = config["max_powers"]

        # loss functions
        self.l1_loss = torch.nn.L1Loss()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        # The rated power loss is calculated in our study only for the mean power output (channel # 1), set the indices below
        self.power_ch = [1]
        # For the zero loss, the zero states are matched for the following channel indices:
        self.zero_ch = [0,1,2,6,7,8] # (our data: 0-2 power, 3-5 wind speed, 6-8 rotor speed)

        self.n_wts = config["n_wts"]
        self.scarce_id = self.n_wts - 1

        # for early stopping
        self.repr_NBMs = config["repr_NBMs"]
        self.mapping_norm_stats = config["mapping_norm_stats"]
    

    def train(self, max_gen_iter, models, tr_dataloaders, target_val_dataloader, optimizers):
        '''
        Main training loop: Trains the StarGAN by performing generator and discriminator updates across all mapping directions.

        Args:
            max_gen_iter (int): Maximunm number of iterations (1 iteration equals mapping a batch from *every* domain to another one) to perform.
            models (dictionary): A dictionary containing the generator and the discriminator
            tr_dataloaders (dictionary): Dictionary containing the training dataloaders of every setup's WT
            target_val_dataloader (torch dl): Dataloader of the target WT's normal validatoin data for early stopping
            optimizers (dictionary): Dictionary containing an optimizer for the generator and one for the discriminator.
        '''
        gen, disc = [models[m] for m in ["gen", "disc"]]
        # EMA for the generator (only), the saved generator will be a moving average state; only for evaluation later
        gen_EMA = AveragedModel(gen, multi_avg_fn=get_ema_multi_avg_fn(0.9995)).to(self.device)
                
        # only the training dataloaders
        # the evaluation for model selection (validation data) and test data are done in separate steps
        tr_dataloaders = [dls["tr"] for dls in tr_dataloaders] 
        tr_iters =  [iter(tr_dl) for tr_dl in tr_dataloaders]

        # optimizers
        opt_G = [optimizers[o] for o in ["opt_G"]][0]
        opt_D = [optimizers[o] for o in ["opt_D"]][0]

        # Early stopping to stop training based on target WT validation data (see paper)
        self.target_val_dl = target_val_dataloader
        stopper = MappingEarlyStopper(warmup_iters=500, patience=1)

        ############
        # TRAINING #
        ############
        
        # for each iteration:
            # *for every WT domain*
                # load a batch of data (named origin data here)
                # randomly set a destination domain to map to

                # for efficiency, the mapping network is updated in both directions:
                # origin->destination and destination->origin; using the same batches

                # calculate the GAN-based and sate consistency-based losses to update the generator and discriminator
                # training is stopped based on rising reconstruction errors from normal, scarce target validation data to source domains 


        for i in range(max_gen_iter+1):  
            for origin_i, dl in enumerate(tr_dataloaders):

                gen.train()
                disc.eval()

                # get batch of data from this *origin* (o) WT
                xo, yo, tr_iters[origin_i] = self.get_inf_batch(tr_iters[origin_i], dl)
                xo, yo = xo.to(self.device), yo.to(self.device) 


                # pick one FIXED random destination domain (d) and then train pair-wise
                destination_wt_id = numpy.random.choice([idx for idx in list(range(0, self.n_wts)) if idx != origin_i])
                xd, yd, tr_iters[destination_wt_id] = self.get_inf_batch(tr_iters[destination_wt_id], tr_dataloaders[destination_wt_id])
                xd, yd = xd.to(self.device), yd.to(self.device)


                ##########################
                # FORWARD PASS GENERATOR #
                ##########################

                # map origin->destination
                O_mapped_to_D = gen(xo, yd) 
                # cycle: map back to origin
                O_cycled_back = gen(O_mapped_to_D, yo) 

                # equivalently for the other direction (destination->origin) too
                D_mapped_to_O = gen(xd, yo) 
                D_cycled_back = gen(D_mapped_to_O, yd) 


                ##########
                # LOSSES #
                ##########

                # for the GAN-QP loss; calculate the discriminator-GAN loss of real and mapped data
                disc_out_real_D, _ = disc(xd)
                disc_out_real_O, _ = disc(xo)

                # also get the classification loss to further determine quality of mapping
                disc_out_fake_D, disc_clsOD = disc(O_mapped_to_D)
                disc_out_fake_O, disc_clsDO = disc(D_mapped_to_O)

                # GAN-QP LOSS for each direction
                gen_loss_OD = torch.mean(disc_out_real_D - disc_out_fake_D)  
                gen_loss_DO = torch.mean(disc_out_real_O - disc_out_fake_O)  

                # generator-based classification loss
                gen_loss_cls_OD = self.cls_loss(disc_clsOD, yd.argmax(dim=-1))
                gen_loss_cls_DO = self.cls_loss(disc_clsDO, yo.argmax(dim=-1))
                

                # cycle-consistency loss
                loss_cyc_o_d_o = self.l1_loss(O_cycled_back, xo) 
                loss_cyc_d_o_d = self.l1_loss(D_cycled_back, xd) 
                cyc_loss = loss_cyc_o_d_o + loss_cyc_d_o_d

                # zero-consistency loss, see paper and _get_zero_loss
                zero_loss_O = self._get_zero_loss(real=xo, generated=O_mapped_to_D, channels=self.zero_ch)
                zero_loss_D = self._get_zero_loss(real=xd, generated=D_mapped_to_O, channels=self.zero_ch)
                zero_loss = zero_loss_O+zero_loss_D


                # rated power consistency, see paper and _get_rated_power_loss
                max_loss_O = self._get_rated_power_loss(real=xo, generated=O_mapped_to_D,
                                                                rated_power_real = self.max_powers[origin_i], rated_power_gen = self.max_powers[destination_wt_id], channels=self.power_ch)
                max_loss_D = self._get_rated_power_loss(real=xd, generated=D_mapped_to_O,
                                                                rated_power_real = self.max_powers[destination_wt_id], rated_power_gen = self.max_powers[origin_i], channels=self.power_ch)
                max_loss = max_loss_O + max_loss_D


                # ANOMALY AUGMENTATION PHASE (see paper appendix):
                # <-> ensure that corrupted data is mapped to corrupted states
                # corrupt the original origin and desination data
                xoC = data_corrupter.corrupt_batch(xo) 
                xdC = data_corrupter.corrupt_batch(xd)

                # cycle a mapping to calculate the cycle-consistency loss
                OC_mapped_to_D = gen(xoC, yd)
                OC_cycled_back = gen(OC_mapped_to_D, yo)
                
                DC_mapped_to_O = gen(xdC, yo)
                DC_cycled_back = gen(DC_mapped_to_O, yd)
                
                # calculate cycle_loss; should return the originally corrupted samples
                loss_cyc_o_d_o_C = self.l1_loss(OC_cycled_back, xoC) 
                loss_cyc_d_o_d_C = self.l1_loss(DC_cycled_back, xdC)
                aug_loss = loss_cyc_o_d_o_C + loss_cyc_d_o_d_C


                ####################################
                #      GENERATOR UPDATE            #
                ####################################

                # GAN-QP generative loss
                gen_loss = gen_loss_OD + gen_loss_DO
                
                # class-predictions loss for the mapped samples
                gen_loss += (self.lambdas["cls_G"] * (gen_loss_cls_OD + gen_loss_cls_DO))

                # The state preserving losses: cycle, zero, and max loss:
                gen_loss += self.lambdas["cyc"] * cyc_loss
                gen_loss += self.lambdas["zero"] * zero_loss
                gen_loss += self.lambdas["max"] * max_loss

                # the cycle-consistency loss for the corrupted batches
                gen_loss += self.lambdas["cyc"] * aug_loss


                ##########################
                # GENERATORS UPDATE STEP #
                ##########################
                opt_G.zero_grad()
                gen_loss.backward()
                opt_G.step()

                # EMA update
                gen_EMA.update_parameters(gen)


                ####################################
                #          CRITIC UPDATES          #
                ####################################
                gen.eval()
                disc.train()
                opt_D.zero_grad()

                disc_loss_O, cls_loss_O = self.get_critic_loss_GANQP(disc=disc, samples_fake=D_mapped_to_O.detach(),
                                        samples_real=xo, labels_real=yo)
                disc_loss_D, cls_loss_D = self.get_critic_loss_GANQP(disc, O_mapped_to_D.detach(), xd, yd)
                
                # adversarial + class
                d_tot_loss = (disc_loss_O + disc_loss_D) + (self.lambdas["cls_D"] * (cls_loss_O + cls_loss_D))


                d_tot_loss.backward()
                opt_D.step()

                gen.train()
                disc.eval()
                
            ############
            # PRINTING #
            ############
            for step in [100, 500, 1000, 2500, 5000, 7500, 10000]: 
                if i == step: print(f"{step} iterations finished.")


            #####################################
            # EARLY STOPPING & GENERATOR SAVING #
            ####################################

            earlystopping_check_every = 500 # evaluate, save, and possibly stop every 500th iteration 
            if (i % earlystopping_check_every == 0) and (i > 500):

                score = mapped_target_avg_score(gen_EMA.module, self.repr_NBMs,
                                                        self.target_val_dl, self.mapping_norm_stats,
                                                            self.scarce_id, self.device) 

                status = stopper.update(i, score)

                if status["is_best"]:
                    self.save_generator(gen_EMA, "_best")
                if status["stop"]:
                    print(f"Early stop @ {i}: through at {stopper.best_iter} (score {stopper.best:.5f})")
                    break 

            gen.train(); disc.eval() # restore training mode after eval pass

        # training loop finish reached
        summary = {
            "best_score": float(stopper.best),
            "best_iter": int(stopper.best_iter) if stopper.best_iter is not None else None,
            "last_iter": int(i),
        }

        with open(self.save_dir.joinpath("es_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        print("Finished all iterations.")
        return True


    def save_generator(self, gen_EMA, suffix):
        save_checkpoint(self.save_dir.joinpath(f"gen{suffix}.pt"), gen_EMA.module, None, None, None)


    def get_critic_loss_GANQP(self, disc, samples_fake, samples_real, labels_real):
        '''
        Calculates the discriminator loss, returns the GAN-QP critic loss and the classifier loss. 
        See paper, StarGAN- and GAN-QP framework.

        Args:
            disc: The discriminator
            sample_fake: fake / generated / mapped samples
            sample_real: A batch of real data
            label_real: The real labels of the real data for the discriminator to learn classifier boundaries
        ''' 
        # real data discriminator output
        f_real, cls_real = disc(samples_real)
        # discriminator output for generated data
        f_fake, _ = disc(samples_fake)         
        
        # GAN-QP disc loss
        disc_loss = f_real - f_fake
        x_norm = 1 * (samples_real - samples_fake).abs().mean()
        disc_loss =  -disc_loss + 0.5 * disc_loss ** 2 / x_norm
        disc_loss = disc_loss.mean()        
        
        # classification loss
        cls_loss = self.cls_loss(cls_real, labels_real.argmax(dim=-1))

        return disc_loss, cls_loss     


    def _get_zero_loss(self, real, generated, channels = []):
        '''
        The zero loss, punishing zero states mapped to non-zero states.
        Due to our normalization adjustment, zero states in the power, windspeed, and rotor variables are normalized to -1.0
        '''

        # create a mask for positions where the provided REAL channels are at zero (roughly, from -0.98 to -1.02)
        mae = torch.tensor(0.0, device=self.device)
        for ch in channels:
            real_ch = real[:, ch, :]
            min_mask = torch.logical_and(real_ch <= -0.99, real_ch >= -1.01)
            # apply mask to fake/mapped/generated data to compare these positions only
            gen_ch = generated[:, ch, :]
            masked_gen = gen_ch[min_mask]
            # calculate mae loss on the masked data
            if min_mask.sum() > 0: mae += torch.mean(torch.abs(masked_gen - -1.0))
            else: mae += torch.tensor(0.0)
        return mae


    def _get_rated_power_loss(self, real, generated, rated_power_real, rated_power_gen, channels = []):
        '''
        Calculates the rated power loss, see paper. Punishes deviations between the rated power across domains. 
        '''

        # calculate a mask for where the power is at the rated capacity in the real sample
        # rated_power_real corresponds to the normalized value of the rated power for this WT
        real_ch = real[:, channels, :]
        max_mask = torch.logical_and(real_ch >= rated_power_real*0.99, real_ch <= rated_power_real * 1.01)
        
        # only consider those positions for the loss in the generated sample resembling the other domain
        gen_ch = generated[:, channels, :]
        masked_gen = gen_ch[max_mask]

        # calculate MAE loss
        # we expect the power to be at the (normalized) rated capacity of the other domain, rated_power_gen, and punish deviations from the generated value
        if max_mask.sum() > 0: mae = torch.mean(torch.abs(masked_gen - rated_power_gen))
        else: mae=torch.tensor(0.0)
        return mae


    def get_inf_batch(self, iterator, dl):
        try: 
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(dl)
            x, y = next(iterator)
        return x, y, iterator