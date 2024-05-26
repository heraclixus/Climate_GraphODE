"""
adapted partly from Nvidia's makani package for computing losses related to SFNO 
"""

from typing import Optional, Tuple, List
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from lib.grids import GridQuadrature
import torch_harmonics as harmonics
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights


# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """
    Computes the Lp loss on the sphere.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        crop_shape : Tuple[int, int] = None,
        crop_offset : Tuple[int, int] = None,
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = False,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
        jacobian: Optional[str] = "s2",
        quadrature_rule: Optional[str] = "naive",
    ):
        super(GeometricLpLoss, self).__init__()

        self.p = p
        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        # get the quadrature
        self.quadrature = GridQuadrature(
            quadrature_rule, img_shape=self.img_shape, crop_shape=self.crop_shape, crop_offset=self.crop_offset, normalize=True, pole_mask=self.pole_mask
        )

    # removed chw 
    def abs(self, prd: torch.Tensor, tar: torch.Tensor):
        num_examples = prd.size()[0]

        all_norms = self.quadrature(torch.abs(prd - tar) ** self.p)
        all_norms = all_norms.reshape(num_examples, -1)

        if not self.squared:
            all_norms = all_norms ** (1.0 / self.p)

        # apply channel weighting
        # all_norms = chw * all_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    # removed chw
    def rel(self, prd: torch.Tensor, tar: torch.Tensor):
        num_examples = prd.size()[0]

        diff_norms = self.quadrature(torch.abs(prd - tar) ** self.p)
        diff_norms = diff_norms.reshape(num_examples, -1)

        tar_norms = self.quadrature(torch.abs(tar) ** self.p)
        tar_norms = tar_norms.reshape(num_examples, -1)

        # divide the ratios
        frac_norms = diff_norms / tar_norms

        if not self.squared:
            frac_norms = frac_norms ** (1.0 / self.p)

        # setup return value
        # retval = chw * frac_norms
        retval = frac_norms

        if self.reduction:
            if self.size_average:
                retval = torch.mean(retval)
            else:
                retval = torch.sum(retval)

        return retval

    # removed chw 
    def forward(self, prd: torch.Tensor, tar: torch.Tensor, vars):
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar)
        loss_dict = {}        
        loss_dict["loss"] = loss
        return loss_dict

# double check if polar optimization has an effect - we use 5 here by default
class GeometricH1Loss(nn.Module):
    """
    Computes the weighted H1 loss on the sphere.
    Alpha is a parameter which balances the respective seminorms.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = False,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        alpha: Optional[float] = 0.5,
    ):
        super(GeometricH1Loss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.alpha = alpha

        self.sht = harmonics.RealSHT(*img_shape, grid="equiangular").float()
        h1_weights = torch.arange(self.sht.lmax).float()
        h1_weights = h1_weights * (h1_weights + 1)
        self.register_buffer("h1_weights", h1_weights)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor):
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            all_norms = self.alpha * torch.sqrt(l2_norm2) + (1 - self.alpha) * torch.sqrt(h1_norm2)
        else:
            all_norms = self.alpha * l2_norm2 + (1 - self.alpha) * h1_norm2

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        tar_coeffs = torch.view_as_real(self.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_l2_norm2 = tar_norm2.reshape(num_examples, -1).sum(dim=-1)
        tar_h1_norm2 = (tar_norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            diff_norms = self.alpha * torch.sqrt(l2_norm2) + (1 - self.alpha) * torch.sqrt(h1_norm2)
            tar_norms = self.alpha * torch.sqrt(tar_l2_norm2) + (1 - self.alpha) * torch.sqrt(tar_h1_norm2)
        else:
            diff_norms = self.alpha * l2_norm2 + (1 - self.alpha) * h1_norm2
            tar_norms = self.alpha * tar_l2_norm2 + (1 - self.alpha) * tar_h1_norm2

        # setup return value
        retval = diff_norms / tar_norms
        if mask is not None:
            retval = retval * mask

        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss
