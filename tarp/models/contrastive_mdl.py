from contextlib import contextmanager
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from copy import deepcopy

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, BCELoss, CELoss, BCEWithLogitsLoss
from tarp.modules.subnetworks import Encoder, Decoder, Predictor
from tarp.modules.recurrent_modules import RecurrentPredictor
from tarp.utils.general_utils import AttrDict, ParamDict, batch_apply, remove_spatial
from tarp.utils.pytorch_utils import pad_seq, ten2ar, ar2ten
from tarp.utils.data_aug import random_shift
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.vis_utils import make_gif_strip, make_image_seq_strip, make_image_strip

class ContrastiveModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)

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
            'detach_reconstruction': True,
            'n_cond_frames': 1,
            'use_seg_mask': False,
            'use_obj_reg': False,
            'detach_seg_mask': True,
            'detach_obj_decoder': True,
            'seg_dec_activation': None,
            'n_class': 1,
            'use_random_rep': False,
            'target_network_update_factor': 0.01,   # percentage of new weights that are carried over
            'target_update_interval': 1,
            'normalization': 'none'
        })

        # Network size
        default_dict.update({
            'img_sz': 64,
            'input_nc': 3,
            'ngf': 8,
            'nz_enc': 128,
            'nz_mid': 128,
            'n_processing_layers': 2,
            'n_pixel_sources': 1,
            'random_shift_prob': 1.,
            'random_shift_pad': 4,
            'anchor_hidden_size': 256
        })

        # Loss weights
        default_dict.update({
            'img_mse_weight': 1.,
            'reward_weights': 1.,
        })
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params


    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.target_encoder = deepcopy(self.encoder)
        self.decoder = Decoder(self._hp)
        if self._hp.anchor_hidden_size is not None:
            self.anchor_mlp = Predictor(self._hp, input_size=self._hp.nz_enc, output_size=self._hp.nz_enc,
                                        spatial=True, num_layers=0, mid_size=self._hp.anchor_hidden_size,
                                        final_activation=None)
        self.anchor_pred =  torch.nn.Linear(self._hp.nz_enc, self._hp.nz_enc, bias=False)

        if self._hp.use_obj_reg:
            self.decoder_obj = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc,
                                                              output_size=1, spatial=False, final_activation=nn.Sigmoid()) for name in self._hp.obj_labels})
        if self._hp.use_seg_mask:
            seg_decoder_hp = deepcopy(self._hp)
            seg_decoder_hp.input_nc = self._hp.n_class
            seg_decoder_hp.dec_last_activation = self._hp.seg_dec_activation
            self.seg_decoder = Decoder(seg_decoder_hp)

    def forward(self, inputs):
        output = AttrDict()

        # encode inputs
        anchor = inputs.images[:, 0]
        positive = inputs.images[:, -1]
        output.enc = self.encoder(anchor) # emb before augmentation

        if self._hp.random_shift_prob > 0:
            anchor = random_shift(
                imgs=anchor,
                pad=self._hp.random_shift_pad,
                prob=self._hp.random_shift_prob
            ).to(self._hp.device)
            positive = random_shift(
                imgs=positive,
                pad=self._hp.random_shift_pad,
                prob=self._hp.random_shift_prob
            ).to(self._hp.device)

        output.anchor_enc = self.encoder(anchor)
        with torch.no_grad():
            output.positive_enc = remove_spatial(self.target_encoder(positive))

        anchor_feat = output.anchor_enc
        if self._hp.anchor_hidden_size is not None:
            anchor_feat = anchor_feat + self.anchor_mlp(anchor_feat)
        output.anchor_pred = self.anchor_pred(remove_spatial(anchor_feat))
        output.logits = torch.matmul(output.anchor_pred, output.positive_enc.T)
        output.logits -= torch.max(output.logits, dim=1, keepdim=True)[0]

        rec_input = output.enc.detach() if self._hp.detach_reconstruction else output.enc
        output.output_imgs = self.decoder(rec_input).images.unsqueeze(1)

        if self._hp.use_seg_mask:
            seg_input = output.enc.detach() if self._hp.detach_seg_mask else output.enc
            output.output_seg = self.seg_decoder(seg_input).images.unsqueeze(1)

        if self._hp.use_obj_reg:
            obj_input = output.enc.detach() if self._hp.detach_obj_decoder else output.enc
            output.obj_labels = AttrDict({name: self.decoder_obj[name](obj_input) for name in self._hp.obj_labels})

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs[:, 0], inputs.images[:, 0])

        if self._hp.use_seg_mask:
            losses.seq_seg_entropy = CELoss()(model_output.output_seg.reshape(self._hp.batch_size, self._hp.n_class, -1),
                                              inputs.seg_targets[:, :1].reshape(self._hp.batch_size, -1))
        if self._hp.use_obj_reg:
            losses.update(AttrDict({name: BCELoss()(model_output.obj_labels[name], inputs.obj_labels[:, :1, i])
                                    for i, name in enumerate(self._hp.obj_labels)}))

        labels = torch.arange(model_output.anchor_enc.shape[0], dtype=torch.long, device=self._hp.device)

        # add valid from done?
        losses.atc_loss = CELoss()(model_output.logits, labels)

        losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)

        if log_images:
            # log predicted images
            img_strip = make_image_strip([inputs.images[:, 0, -min(int(self._hp.input_nc//self._hp.n_frames), 3):], model_output.output_imgs[:, 0, -min(int(self._hp.input_nc//self._hp.n_frames), 3):]])
            self._logger.log_images(img_strip[None], 'generation', step, phase)

            if self._hp.use_seg_mask:
                output_labels = torch.argmax(model_output.output_seg.squeeze(1), dim=1)
                b, _, nc, h, w = model_output.output_seg.shape
                pred_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                gt_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                for c in range(nc):
                    pred_seg[output_labels==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                    gt_seg[inputs.seg_targets[:, :1].squeeze(1).squeeze(1)==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                mask_strip = make_image_strip([(inputs.images[:, 0, -min(int(self._hp.input_nc//self._hp.n_frames), 3):].repeat(1, 3, 1, 1)+1)*255/2,
                                                gt_seg.permute((0, 3, 1, 2)), pred_seg.permute((0, 3, 1, 2))])
                self._logger.log_images(mask_strip[None], 'segmentation', step, phase)

            # attention mask
            self._log_attention_mask(inputs, step, phase)

        labels = torch.arange(model_output.anchor_enc.shape[0], dtype=torch.long, device=self._hp.device)
        correct = torch.argmax(model_output.logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct.float())
        self._logger.log_scalar(accuracy, 'atc_accuracy', step, phase)


    def after_update(self, step):
        if step % self._hp.target_update_interval == 0:
            self._soft_update_target_network(self.target_encoder, self.encoder)

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    @property
    def resolution(self):
        return self._hp.img_sz

