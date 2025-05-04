import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def rld_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand, alpha_temperature):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # scd loss
    student_gt_mask = _get_gt_mask(logits_student, target)
    student_other_mask = _get_other_mask(logits_student, target)
    max_index = torch.argmax(logits_teacher, dim=1)
    teacher_max_mask = _get_gt_mask(logits_teacher, max_index)
    teacher_other_mask = _get_other_mask(logits_teacher, max_index)
    pred_student = F.softmax(logits_student / alpha_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / alpha_temperature, dim=1)
    pred_student = cat_mask(pred_student, student_gt_mask, student_other_mask)
    pred_teacher = cat_mask(pred_teacher, teacher_max_mask, teacher_other_mask)
    log_pred_student = torch.log(pred_student)
    scd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (alpha_temperature**2)

    # mcd loss
    mask = _get_ge_mask(logits_teacher, target)
    assert mask.shape == logits_student.shape
    masked_student = (logits_student / temperature).masked_fill(mask, -1e9)
    log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
    masked_teacher = (logits_teacher / temperature).masked_fill(mask, -1e9)
    pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
    mcd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)

    return alpha * scd_loss + beta * mcd_loss


def multi_rld_loss(out_s_multi, out_t_multi, target, alpha, beta, temperature, logit_stand, alpha_temperature):
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
        
        # Calculate RLD loss components
        # scd loss for multi-scale
        student_gt_mask = _get_gt_mask(out_s, target_r)
        student_other_mask = _get_other_mask(out_s, target_r)
        max_index = torch.argmax(out_t, dim=1)
        teacher_max_mask = _get_gt_mask(out_t, max_index)
        teacher_other_mask = _get_other_mask(out_t, max_index)
        
        pred_student = F.softmax(out_s / alpha_temperature, dim=1)
        pred_teacher = F.softmax(out_t / alpha_temperature, dim=1)
        pred_student = cat_mask(pred_student, student_gt_mask, student_other_mask)
        pred_teacher = cat_mask(pred_teacher, teacher_max_mask, teacher_other_mask)
        log_pred_student = torch.log(pred_student)
        
        # Calculate SCD loss for each local region
        scd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='none') * (alpha_temperature**2)
        scd_loss = torch.sum(scd_loss, dim=1)  # Sum over class dimension
        
        # mcd loss for multi-scale
        mask = _get_ge_mask(out_t, target_r)
        assert mask.shape == out_s.shape
        masked_student = (out_s / temperature).masked_fill(mask, -1e9)
        log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
        masked_teacher = (out_t / temperature).masked_fill(mask, -1e9)
        pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
        
        # Calculate MCD loss for each local region
        mcd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none') * (temperature**2)
        mcd_loss = torch.sum(mcd_loss, dim=1)  # Sum over class dimension
        
        # Calculate the combined base loss
        base_loss = alpha * scd_loss + beta * mcd_loss
        
        # Calculate masks for different prediction scenarios
        out_t_predict = torch.argmax(out_t, dim=1)
        global_prediction = out_t_predict[:len(target)]

        # Create global prediction masks
        global_prediction_true_mask = (global_prediction == target)
        global_prediction_false_mask = (global_prediction != target)

        global_prediction_true_mask_repeat = global_prediction_true_mask.to(device).repeat(out_t_multi.shape[2])
        global_prediction_false_mask_repeat = global_prediction_false_mask.to(device).repeat(out_t_multi.shape[2])
        
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
        weights = torch.zeros_like(base_loss, device=device)
        weights[gw_lw] = 1.0  # Global Wrong, Local Wrong
        weights[gt_lt] = 1.0  # Global True, Local True
        weights[gw_lt] = 2.0  # Global Wrong, Local True
        weights[gt_lw] = 2.0  # Global True, Local Wrong
        
        # Calculate total loss with weights
        loss = torch.sum(base_loss * weights) / target_r.shape[0]
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("inf or nan detected in multi_rld_loss")
            loss = torch.zeros(1, dtype=torch.float, device=device)
        
        return loss
    except Exception as e:
        print(f"Error in multi_rld_loss: {str(e)}")
        return torch.zeros(1, device=device)


def _get_ge_mask(logits, target):
    """Get mask for logits >= target_logit."""
    if logits.dim() != 2 or target.dim() != 1 or logits.size(0) != target.size(0):
        return torch.zeros_like(logits).bool()
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits >= gt_value, 1, 0).bool()
    return mask

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class SDD_RLD(Distiller):
    """Scale Decoupled Refined Logit Distillation"""

    def __init__(self, student, teacher, cfg):
        super(SDD_RLD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.RLD.CE_WEIGHT
        self.alpha = cfg.RLD.ALPHA
        self.beta = cfg.RLD.BETA
        self.temperature = cfg.RLD.T
        self.warmup = cfg.RLD.WARMUP
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.alpha_temperature = cfg.RLD.ALPHA_T
        self.M = cfg.M if hasattr(cfg, 'M') else '[1]'

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        warmup_factor = min(kwargs.get("epoch", 1) / self.warmup, 1.0)
        
        if self.M == '[1]':
            loss_rld = warmup_factor * rld_loss(
                logits_student,
                logits_teacher,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.logit_stand,
                self.alpha_temperature,
            )
        else:
            loss_rld = warmup_factor * multi_rld_loss(
                patch_s,
                patch_t,
                target,
                self.alpha,
                self.beta,
                self.temperature,
                self.logit_stand,
                self.alpha_temperature,
            )
            
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_rld,
        }
        return logits_student, losses_dict