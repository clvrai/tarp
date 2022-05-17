from contextlib import contextmanager
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from copy import deepcopy

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, BCELoss, CELoss, BCEWithLogitsLoss
from tarp.modules.subnetworks import Encoder, Decoder, Predictor
from tarp.modules.recurrent_modules import RecurrentPredictor
from tarp.utils.general_utils import AttrDict, ParamDict, batch_apply
from tarp.utils.pytorch_utils import pad_seq, ar2ten, ten2ar
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.vis_utils import make_gif_strip, make_image_seq_strip, make_image_strip

class TARPValueModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self._task_names = [task_name().name if not isinstance(task_name, str) else task_name
                              for task_name in self._hp.task_names]

        self.build_network()

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass

    def _default_hparams(self):
        default_dict = ParamDict({
            'use_skips': False,
            'skips_stride': 2,
            'add_weighted_pixel_copy': False, # if True, adds pixel copying stream for decoder
            'pixel_shift_decoder': False,
            'use_convs': True,
            'use_custom_convs': False,
            'detach_reconstruction': True,
            'detach_discounted_return_heads': False,
            'n_cond_frames': 1,
            'use_seg_mask': False,
            'detach_seg_mask': True,
            'seg_dec_activation': None,
            'n_class': 1,
            'use_random_rep': False,
            'use_obj_labels': False,
            'normalization': 'none',
        })

        # Network size
        default_dict.update({
            'img_sz': 32,
            'input_nc': 3,
            'ngf': 8,
            'nz_enc': 32,
            'nz_mid': 32,
            'n_processing_layers': 3,
            'n_pixel_sources': 1,
        })

        # Loss weights
        default_dict.update({
            'img_mse_weight': 1.,
            'reward_weights': 1.,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.decoder = Decoder(self._hp)
        if self._hp.use_obj_labels:
            self.decoder_obj = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc,
                                                              output_size=1, spatial=False, final_activation=nn.Sigmoid()) for name in self._hp.obj_labels})
        if self._hp.use_seg_mask:
            seg_decoder_hp = deepcopy(self._hp)
            seg_decoder_hp.input_nc = self._hp.n_class
            seg_decoder_hp.dec_last_activation = self._hp.seg_dec_activation
            self.seg_decoder = Decoder(seg_decoder_hp)

        self.discounted_return_heads = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc,
                                                                      output_size=1, spatial=False) for name in self._task_names})

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # encode inputs
        enc = self.encoder(inputs.images[:, 0])
        output.update({'pred': enc, 'rec_input': enc})

        if self._hp.use_random_rep:
            output.pred = torch.rand(output.pred.shape, device=self._hp.device)

        rec_input = output.rec_input.detach() if self._hp.detach_reconstruction else output.rec_input
        output.output_imgs = self.decoder(rec_input).images.unsqueeze(1)
        if self._hp.use_obj_labels:
            output.obj_labels = AttrDict({name: self.decoder_obj[name](rec_input) for name in self._hp.obj_labels})

        if self._hp.use_seg_mask:
            seg_input = output.pred.detach() if self._hp.detach_seg_mask else output.pred
            output.output_seg = self.seg_decoder(seg_input).images.unsqueeze(1)


        # reward decoding
        discounted_return_input = output.pred.detach() if self._hp.detach_discounted_return_heads else output.pred
        output.discounted_returns = AttrDict({name: self.discounted_return_heads[name](discounted_return_input)
                                   for name in self._task_names})

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # image reconstruction loss
        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs,
                                                             inputs.images[:, :1])
        if self._hp.use_seg_mask:
            losses.seq_seg_entropy = CELoss()(model_output.output_seg.reshape(self._hp.batch_size, self._hp.n_class, -1),
                                              inputs.seg_targets[:, :1].reshape(self._hp.batch_size, -1))

        if self._hp.use_obj_labels:
            losses.update(AttrDict({name: BCELoss()(model_output.obj_labels[name], inputs.obj_labels[:, :, i])
                                    for i, name in enumerate(self._hp.obj_labels)}))

        # reward regression loss
        losses.update(AttrDict({name: L2Loss(self._hp.reward_weights)(model_output.discounted_returns[name][inputs.task_id==i],
                                                                      inputs.discounted_returns[:, :1][inputs.task_id==i])
                                for i, name in enumerate(self._task_names)}))
        losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        if log_images:
            # log predicted images
            img_strip = make_image_strip([inputs.images[:, 0, -int(self._hp.input_nc//self._hp.n_frames):],
                                          model_output.output_imgs[:, 0, -int(self._hp.input_nc//self._hp.n_frames):]])
            self._logger.log_images(img_strip[None], 'generation', step, phase)

            if self._hp.use_seg_mask:
                output_labels = torch.argmax(model_output.output_seg.squeeze(1), dim=1)
                b, _, nc, h, w = model_output.output_seg.shape
                pred_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                gt_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                for c in range(nc):
                    pred_seg[output_labels==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                    gt_seg[inputs.seg_targets[:, :1].squeeze(1).squeeze(1)==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                input_images = inputs.images[:, 0, -int(self._hp.input_nc//self._hp.n_frames):]
                if input_images.shape[1] == 1:
                    input_images = input_images.repeat((1, 3, 1, 1))
                mask_strip = make_image_strip([(input_images+1)*255/2,
                                                gt_seg.permute((0, 3, 1, 2)), pred_seg.permute((0, 3, 1, 2))])
                self._logger.log_images(mask_strip[None], 'segmentation', step, phase)

            # attention mask
            # self._log_attention_mask(inputs, step, phase)

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    @property
    def resolution(self):
        return self._hp.img_sz


