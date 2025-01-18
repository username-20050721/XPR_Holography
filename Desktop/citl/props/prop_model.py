"""
Parameterized propagations

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
"""
import torch
import torch.nn as nn
from unet import UnetGenerator, init_weights
import props.prop_ideal as prop_ideal
from props.prop_submodules import Field2Input, Output2Field, create_gaussian
from wtconv import WTConv2d
class CNNpropCNN(nn.Module):
    def __init__(self, prop_dist, wavelength, feature_size, prop_type='ASM', F_aperture=1.0,
                 prop_dists_from_wrp=None, linear_conv=True, slm_res=(2160, 3840),
                 num_downs_slm=5, num_feats_slm_min=32, num_feats_slm_max=128,
                 num_downs_target=5, num_feats_target_min=32, num_feats_target_max=128,
                 norm=nn.InstanceNorm2d, slm_coord='both', target_coord='both',f_amp_init_th=0.36,
                 use_wt = False
                ):
        super(CNNpropCNN, self).__init__()
        
        ############
        # Learned Parameters
        self.slm_latent_amp = nn.Parameter(torch.ones(1, 1, *slm_res, requires_grad=True))
        self.slm_latent_phase = nn.Parameter(torch.zeros(1, 1, *slm_res, requires_grad=True))
        fourier_res = tuple([2 * p for p in slm_res])
        init_f_amp = create_gaussian(fourier_res, sigma=2e-4/wavelength).reshape(1, 1, *fourier_res)
        init_f_amp = (init_f_amp / init_f_amp.max()).clamp(0., f_amp_init_th) / f_amp_init_th
        self.f_latent_amp = nn.Parameter(init_f_amp.detach().requires_grad_(True), requires_grad=True)
        self.f_latent_phase = nn.Parameter(torch.zeros(1, 1, *fourier_res, requires_grad=True))

        ##################
        # Model pipeline #
        ##################
        # SLM Network
        slm_cnns = []
        slm_cnn_res = tuple(res if res % (2 ** num_downs_slm) == 0 else
                            res + (2 ** num_downs_slm - res % (2 ** num_downs_slm))
                            for res in slm_res)
        slm_input = Field2Input(slm_cnn_res, coord=slm_coord,
                                latent_amp=self.slm_latent_amp, latent_phase=self.slm_latent_phase)
        slm_cnns += [slm_input]
        if not use_wt:
            slm_cnn = UnetGenerator(input_nc=4, output_nc=2,
                                    num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                    max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
            init_weights(slm_cnn, init_type='normal')
            slm_cnns += [slm_cnn]
        else:
            slm_cnn1 =  WTConv2d(in_channels = 4, out_channels = 4, wt_levels = 4)
            slm_cnn_out = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, padding=2)
            init_weights(slm_cnn1, init_type='normal')
            init_weights(slm_cnn_out, init_type='normal')
            slm_cnns += [slm_cnn1, slm_cnn_out]
        slm_output = Output2Field(slm_res, slm_coord)
        slm_cnns += [slm_output]
        self.slm_cnn = nn.Sequential(*slm_cnns)

        # Propagation from the SLM plane to the WRP.
        self.prop_slm_wrp = prop_ideal.Propagation(prop_dist, wavelength, feature_size,
                                                   prop_type=prop_type, linear_conv=linear_conv,
                                                   F_aperture=F_aperture, learned_amp=self.f_latent_amp,
                                                   learned_phase=self.f_latent_phase)

        # Propagation from the WRP to other planes.
        self.f_latent_amp_wrp = self.f_latent_amp
        self.f_latent_phase_wrp = self.f_latent_phase
        #self.f_latent_amp_wrp = nn.Parameter(torch.ones(1, len(prop_dists_from_wrp),
        #                                    *fourier_res, requires_grad=True))
        #self.f_latent_phase_wrp = nn.Parameter(torch.zeros(1, len(prop_dists_from_wrp),
        #                                      *fourier_res, requires_grad=True))
        self.prop_wrp_target = prop_ideal.Propagation(prop_dists_from_wrp, wavelength, feature_size,
                                                      prop_type=prop_type,
                                                      linear_conv=linear_conv,
                                                      F_aperture=F_aperture,
                                                      learned_amp=self.f_latent_amp_wrp,
                                                      learned_phase=self.f_latent_phase_wrp)

        # Target network (This is either included (prop later) or not (prop before, which is then basically NH3D).
        target_cnn_res = tuple(res if res % (2 ** num_downs_target) == 0 else
                               res + (2 ** num_downs_target - res % (2 ** num_downs_target)) for res in slm_res)
        target_input = Field2Input(target_cnn_res, coord=target_coord, shared_cnn=True)

        if not use_wt:
            target_cnn = UnetGenerator(input_nc=4, output_nc=2,
                                       num_downs=num_downs_target, nf0=num_feats_target_min,
                                       max_channels=num_feats_target_max, norm_layer=norm, outer_skip=True)
            init_weights(target_cnn, init_type='normal')
            # shared target cnn requires permutation in channels here.
            target_output = Output2Field(slm_res, target_coord, num_ch_output=1)
            target_cnns = [target_input, target_cnn, target_output]
        else:
            target_cnn1 =  WTConv2d(in_channels = 4, out_channels = 4, wt_levels = 4)
            target_cnn_out = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=5, padding=2)
            init_weights(target_cnn1, init_type='normal')
            init_weights(target_cnn_out, init_type='normal')
            target_output = Output2Field(slm_res, target_coord, num_ch_output=1)
            target_cnns = [target_input, target_cnn1, target_cnn_out, target_output]

        self.target_cnn = nn.Sequential(*target_cnns)

    def forward(self, field):
        slm_field = self.slm_cnn(field)  # Applying CNN at SLM plane
    
        wrp_field = self.prop_slm_wrp(slm_field)  # Propagation from SLM to Intermediate plane. 2g
    
        target_field = self.prop_wrp_target(wrp_field)  # Propagation from Intermediate plane to Target planes. 8g
    
        amp = self.target_cnn(target_field).abs().squeeze()  # Applying CNN at Target planes. 25g
        phase = target_field.angle().squeeze() #0g
        output_field = amp * torch.exp(1j * phase)#1g
        return output_field

    def epoch_end_images(self, prefix):
        """
        execute at the end of epochs

        :param prefix:
        :return:
        """
        #################
        # Reconstructions
        logger = self.logger.experiment
        recon_amp = self.recon_amp[prefix][0]
        target_amp = self.target_amp[prefix][0]
        # some visualizations - we now know how it would go on so don't do it and instead use visualize_model.py
        for i in range(recon_amp.shape[0]):
            logger.add_image(f'amp_recon/{prefix}_{i}', (recon_amp[i:i+1, ...]).clip(0, 1), self.global_step)
            logger.add_image(f'amp_target/{prefix}_{i}', target_amp[i:i+1, ...].clip(0, 1), self.global_step)
        ############
        # Parameters
        if self.lut is not None:
            self.plot_lut(prefix)
        if self.slm_latent_amp is not None:
            logger.add_image(f'slm_latent_amp/{prefix}',
                         ((self.slm_latent_amp - self.slm_latent_amp.min()) /
                          (self.slm_latent_amp.max() - self.slm_latent_amp.min())).squeeze(0), self.global_step)
        if self.slm_latent_phase is not None:
            logger.add_image(f'slm_latent_phase/{prefix}',
                         ((self.slm_latent_phase - self.slm_latent_phase.min()) /
                          (self.slm_latent_phase.max() - self.slm_latent_phase.min())).squeeze(0), self.global_step)
        if self.f_latent_amp is not None:
            logger.add_image(f'f_latent_amp/{prefix}',
                         ((self.f_latent_amp - self.f_latent_amp.min()) /
                          (self.f_latent_amp.max() - self.f_latent_amp.min())).squeeze(0), self.global_step)
        if self.f_latent_phase is not None:
            logger.add_image(f'f_latent_phase/{prefix}',
                         ((self.f_latent_phase - self.f_latent_phase.min()) /
                          (self.f_latent_phase.max() - self.f_latent_phase.min())).squeeze(0), self.global_step)
        if self.f_latent_amp_wrp is not None:
            for i in range(self.f_latent_amp_wrp.shape[1]):
                f_latent_amp_wrp = self.f_latent_amp_wrp[:, i, ...]
                logger.add_image(f'f_latent_amp_wrp/{prefix}_{i}',
                             ((f_latent_amp_wrp - f_latent_amp_wrp.min()) /
                              (f_latent_amp_wrp.max() - f_latent_amp_wrp.min())), self.global_step)
        if self.f_latent_phase_wrp is not None:
            for i in range(self.f_latent_phase_wrp.shape[1]):
                f_latent_phase_wrp = self.f_latent_phase_wrp[:, i, ...]
                logger.add_image(f'f_latent_phase_wrp/{prefix}_{i}',
                             ((f_latent_phase_wrp - f_latent_phase_wrp.min()) /
                              (f_latent_phase_wrp.max() - f_latent_phase_wrp.min())), self.global_step)

'''
    def preload_H(self):
        """
        premultiply kernels at fourier plane
        :return:
        """
        if self.prop_slm_wrp is not None:
            self.prop_slm_wrp.preload_H()
        if self.prop_wrp_target is not None:
            self.prop_wrp_target.preload_H()

    @property
    def plane_idx(self):
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        """

        """
        if idx is None:
            return
        self._plane_idx = idx
        if self.prop_wrp_target is not None and len(self.prop_wrp_target) > 1:
            self.prop_wrp_target.plane_idx = idx
        if self.f_latent_amp_wrp is not None and self.f_latent_amp_wrp.shape[1] > 1:
            self.f_latent_amp_wrp = nn.Parameter(self.f_latent_amp_wrp[:, idx:idx+1, ...],
                                                 requires_grad=False)
        if self.f_latent_phase_wrp is not None and self.f_latent_phase_wrp.shape[1] > 1:
            self.f_latent_phase_wrp = nn.Parameter(self.f_latent_phase_wrp[:, idx:idx+1, ...],
                                                   requires_grad=False)
        if self.target_cnn is not None and self.target_cnn[-1].num_ch_output > 1:
            self.target_cnn[-1].num_ch_output = 1
'''