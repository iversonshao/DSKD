import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
#from .DKD import dkd_loss as dkd_loss_origin

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True) 
    return (logit - mean) / (1e-7 + stdv)

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, logit_stand=False):

    if logit_stand:
        logits_student = normalize(logits_student)
        logits_teacher = normalize(logits_teacher)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    #size_average=False
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (temperature**2)
        #/ target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    #size_average=False
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)
        #/ target.shape[0]
    )
    # tckd_loss = torch.sum(tckd_loss, dim=1)
    # nckd_loss = torch.sum(nckd_loss, dim=1)
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# def compute_dkd_loss(logits_student, logits_teacher, target, temperature):
#     """Compute basic DKD loss without alpha and beta weights."""
#     gt_mask = _get_gt_mask(logits_student, target)
#     other_mask = _get_other_mask(logits_student, target)
    
#     # Teacher-class KD loss
#     pred_student = F.softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     pred_student = cat_mask(pred_student, gt_mask, other_mask)
#     pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
#     log_pred_student = torch.log(pred_student)
    
#     tckd_loss = (
#         F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
#         * (temperature ** 2)
#         / target.shape[0]
#     )
    
#     # Non-ground-truth class KD loss
#     pred_teacher_part2 = F.softmax(
#         logits_teacher / temperature - 1000.0 * gt_mask, dim=1
#     )
#     log_pred_student_part2 = F.log_softmax(
#         logits_student / temperature - 1000.0 * gt_mask, dim=1
#     )
    
#     nckd_loss = (
#         F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
#         * (temperature ** 2)
#         / target.shape[0]
#     )
    
#     # Sum over class dimension
#     tckd_loss = torch.sum(tckd_loss, dim=1)
#     nckd_loss = torch.sum(nckd_loss, dim=1)
    
#     return tckd_loss, nckd_loss

def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature, logit_stand=False):
    device = out_t_multi.device
    try:
        # Shape: B X C X N to N*B X C
        out_s_multi_t = out_s_multi.permute(2, 0, 1)
        out_t_multi_t = out_t_multi.permute(2, 0, 1)
        
        out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
        out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
        
        # Apply logit standardization if enabled
        if logit_stand:
            out_t = normalize(out_t)
            out_s = normalize(out_s)
            
        target_r = target.repeat(out_t_multi.shape[2])
        gt_mask = _get_gt_mask(out_s, target_r)
        other_mask = _get_other_mask(out_s, target_r)
        #loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature, False)  # False since already normalized
        # Compute basic DKD loss

        pred_student = F.softmax(out_s / temperature, dim=1)
        pred_teacher = F.softmax(out_t / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='none') * (temperature ** 2)
            #/ target_r.shape[0]
        )
        pred_teacher_part2 = F.softmax(out_t / temperature - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(out_s / temperature - 1000.0 * gt_mask, dim=1)
        
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none') * (temperature ** 2)
            #/ target_r.shape[0]
        )
        tckd_loss = torch.sum(tckd_loss, dim=1)
        nckd_loss = torch.sum(nckd_loss, dim=1)
        base_loss = alpha * tckd_loss + beta * nckd_loss        
        # Calculate masks for different prediction scenarios``
        out_t_predict = torch.argmax(out_t, dim=1)
        global_prediction = out_t_predict[:len(target)]

        # Create global prediction masks
        global_prediction_true_mask = (global_prediction == target)
        global_prediction_false_mask = (global_prediction != target)

        global_prediction_true_mask_repeat = global_prediction_true_mask.to(device).repeat(out_t_multi.shape[2])
        global_prediction_false_mask_repeat = global_prediction_false_mask.to(device).repeat(out_t_multi.shape[2])
        # Weight different scenarios
        # mask_false[global_prediction_false_mask_repeat] = False
        # mask_false[0:len(target)] = False
        # gt_lw = mask_false

        # mask_true[global_prediction_true_mask_repeat] = False
        # mask_true[0:len(target)] = False
        # gw_lt = mask_true

        # mask_false = out_t_predict != target_r
        # mask_true = out_t_predict == target_r
        
        # index = torch.zeros_like(loss).float()
        
        # mask_false[global_prediction_true_mask_repeat] = False
        # gw_lw = mask_false
        
        # mask_true[global_prediction_false_mask_repeat] = False
        # gt_lt = mask_true
        # Initialize local prediction masks
        local_true = (out_t_predict == target_r)
        local_false = (out_t_predict != target_r)
        
        # Compute different scenarios
        gt_lw = local_false.clone()  # Global True, Local Wrong
        gt_lw[global_prediction_false_mask_repeat] = False
        gt_lw[:len(target)] = False
        
        gw_lt = local_true.clone()  # Global Wrong, Local True
        gw_lt[global_prediction_true_mask_repeat] = False
        gw_lt[:len(target)] = False
        
        gw_lw = local_false.clone()  # Global Wrong, Local Wrong
        gw_lw[global_prediction_true_mask_repeat] = False
        
        gt_lt = local_true.clone()  # Global True, Local True
        gt_lt[global_prediction_false_mask_repeat] = False
        # Apply weights
        #weights = torch.zeros_like(base_loss, device=device)
        weights = torch.zeros(out_t.size(0), dtype=torch.float, device=device)
        weights[gw_lw] = 1.0  # Global Wrong, Local Wrong
        weights[gt_lt] = 1.0  # Global True, Local True
        weights[gw_lt] = 2.0  # Global Wrong, Local True
        weights[gt_lw] = 2.0  # Global True, Local Wrong     
        # Weight assignment
        
        num_valid = weights.sum()
        if num_valid > 0:
            loss = base_loss * (num_valid / weights.numel())
        else:
            loss = base_loss
        #loss = torch.sum(base_loss * weights)
        return loss
    except Exception as e:
        print(f"Error in multi_dkd: {str(e)}")
        return torch.zeros(1, device=device)

class SDD_DKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(SDD_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.M = cfg.M if hasattr(cfg, 'M') else [1]
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        if self.M == '[1]':
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.logit_stand,
            )

        else:
            loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * multi_dkd(
                patch_s,
                patch_t,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.logit_stand,
            )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict