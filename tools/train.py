import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
import ast
import json

verify_dict = {
    "res32x4": "resnet32x4",
    "res8x4": "resnet8x4",
    "shuv2": "ShuffleV2",
    "res18": "ResNet18",
    "res34": "ResNet34",
    "res50": "ResNet50",
    "mv1": "MobileNetV1",
    "mv2": "MobileNetV2",
    "res56": "resnet56",
    "res20": "resnet20",
    "res110": "resnet110",
    "res32": "resnet32",
    "vgg13": "vgg13",
    "vgg8": "vgg8",
    "wrn_40_2": "wrn_40_2",
    "wrn_16_2": "wrn_16_2",
    "wrn_40_1": "wrn_40_1",
    "dkd": "DKD",
    "sdd_dkd": "SDD_DKD",
    "kd": "KD",
    "sdd_kd": "SDD_KD",
    "rc": "RC",
    "mlkd": "MLKD",
    "mlkd_noaug": "MLKD_NOAUG",
    "rld": "RLD",
    "sdd_rld": "SDD_RLD",    
    "vanilla": "NONE",
    "swap": "SWAP",
    "revision": "REVISION",
    "res32x4_sdd": "resnet32x4_sdd",
    "res8x4_sdd": "resnet8x4_sdd",
    "shuv2_sdd": "ShuffleV2_sdd",
    "res18_sdd": "ResNet18_sdd",
    "res34_sdd": "ResNet34_sdd",
    "res50_sdd": "ResNet50_sdd",
    "mv1_sdd": "MobileNetV1_sdd",
    "mv2_sdd": "MobileNetV2_sdd",
    "res56_sdd": "resnet56_sdd",
    "res20_sdd": "resnet20_sdd",
    "res110_sdd": "resnet110_sdd",
    "res32_sdd": "resnet32_sdd",
    "vgg13_sdd": "vgg13_sdd",
    "vgg8_sdd": "vgg8_sdd",
    "wrn_40_2_sdd": "wrn_40_2_sdd",
    "wrn_16_2_sdd": "wrn_16_2_sdd",
    "wrn_40_1_sdd": "wrn_40_1_sdd",    
}

def get_full_model_name(short_name, suffix=""):
    if short_name in verify_dict:
        return verify_dict[short_name] + suffix
    return None

def main(cfg, resume, opts):
    tags = cfg.EXPERIMENT.TAG.split(",")
    print(f"Tags: {tags}")  # Debug info

    type_name = tags[0]
    teacher_name = tags[1]
    student_name = tags[2]
    use_sdd_suffix = (args.M != '[1]') or ('sdd' in cfg.DISTILLER.TEACHER) or ('sdd' in cfg.DISTILLER.STUDENT)
    print(use_sdd_suffix)

    reverse_mapping = {
        'res32x4': 'resnet32x4',
        'res8x4': 'resnet8x4',
        'shuv2': 'ShuffleV2',
        'res18': 'ResNet18',
        'res34': 'ResNet34',
        'res50': 'ResNet50',
        'mv1': 'MobileNetV1',
        'mv2': 'MobileNetV2',
        'res56': 'resnet56',
        'res20': 'resnet20',
        'res110': 'resnet110',
        'res32': 'resnet32',
    }

    teacher_key = teacher_name
    if teacher_name in reverse_mapping:
        teacher_key = reverse_mapping[teacher_name]
    if use_sdd_suffix:
        model_key = f"{teacher_key}_sdd"
        verify_key = f"{teacher_name}_sdd"
    else:
        model_key = teacher_key
        verify_key = teacher_name

    student_key = student_name
    if student_name in reverse_mapping:
        student_key = reverse_mapping[student_name]
    if use_sdd_suffix:
        student_model_key = f"{student_key}_sdd"
        student_verify_key = f"{student_name}_sdd"
    else:
        student_model_key = student_key
        student_verify_key = student_name

    # debug print
    print(f"Teacher key: {teacher_key}")
    print(f"Config teacher: {cfg.DISTILLER.TEACHER}")
    print(f"Student key: {student_key}")
    print(f"Config student: {cfg.DISTILLER.STUDENT}")
    assert verify_dict[type_name] == cfg.DISTILLER.TYPE
    assert verify_dict[verify_key] == cfg.DISTILLER.TEACHER
    assert verify_dict[student_verify_key] == cfg.DISTILLER.STUDENT
    if cfg.DATASET.TYPE == "cifar100":
        if any(name in student_name for name in ["shuv1", "shuv2", "mv2"]):
            assert cfg.SOLVER.LR == 0.01
        else:
            assert cfg.SOLVER.LR == 0.05

    base_dir = os.path.join(cfg.EXPERIMENT.PROJECT, ",".join([teacher_name, student_name]), type_name)
    

    remaining_tags = tags[3:]
    has_stand = 'stand' in remaining_tags
    
    if has_stand:
        remaining_tags = [tag for tag in remaining_tags if tag != 'stand']
        if remaining_tags:
            experiment_name_for_output = os.path.join(base_dir, "stand", ",".join(remaining_tags))
        else:
            experiment_name_for_output = os.path.join(base_dir, "stand")
    else:
        if remaining_tags:
            experiment_name_for_output = os.path.join(base_dir, ",".join(remaining_tags))
        else:
            experiment_name_for_output = os.path.join(base_dir, "base")
    
    print(f"Output directory: {experiment_name_for_output}")
    
    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True, M=cfg.M)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False, M=cfg.M)
        else:
            if use_sdd_suffix:
                net, pretrain_model_path = cifar_model_dict[model_key]
                assert pretrain_model_path is not None
                
                print(f"Using SDD models with M={cfg.M}")
                model_teacher = net(num_classes=num_classes, M=cfg.M)
                model_student = cifar_model_dict[student_model_key][0](
                    num_classes=num_classes, M=cfg.M
                )
            else:
                net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
                assert pretrain_model_path is not None
                
                model_teacher = net(num_classes=num_classes)
                model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name_for_output, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)

# for cifar100
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--same-t", action="store_true")
    parser.add_argument("--base-temp", type=float, default=2.)
    parser.add_argument("--kd-weight", type=float, default=9.)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--gpu", default=0, help="GPU id to use")
    parser.add_argument("--M", type=str, default='[1]', help="scale levels for SDD")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.M = args.M.strip("'\"")
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.aug:
        cfg.EXPERIMENT.AUG = True
        cfg.EXPERIMENT.TAG += ',aug'

    if args.logit_stand and cfg.DISTILLER.TYPE in ['KD', 'SDD_KD', 'DKD', 'SDD_DKD','MLKD', 'RLD', 'SDD_RLD', 'SWAP', 'RC', 'MLKD_NOAUG']:
        cfg.EXPERIMENT.LOGIT_STAND = True
        cfg.EXPERIMENT.TAG += ',stand'
        if cfg.DISTILLER.TYPE == 'KD' or cfg.DISTILLER.TYPE == 'SDD_KD' or cfg.DISTILLER.TYPE == 'SWAP' or cfg.DISTILLER.TYPE == 'MLKD' or cfg.DISTILLER.TYPE == 'MLKD_NOAUG':
            cfg.KD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD' or cfg.DISTILLER.TYPE == 'SDD_DKD':
            cfg.DKD.ALPHA = cfg.DKD.ALPHA * args.kd_weight
            cfg.DKD.BETA = cfg.DKD.BETA * args.kd_weight
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'RLD'or cfg.DISTILLER.TYPE == 'SDD_RLD':
            cfg.RLD.ALPHA = cfg.RLD.ALPHA * args.kd_weight
            cfg.RLD.BETA = cfg.RLD.BETA * args.kd_weight
            cfg.RLD.T = args.base_temp
            cfg.RLD.ALPHA_T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'RC':
            cfg.RC.KD_WEIGHT = cfg.RC.KD_WEIGHT * args.kd_weight
            cfg.RC.RC_WEIGHT = cfg.RC.RC_WEIGHT * args.kd_weight
            cfg.RC.T = args.base_temp
    
    if cfg.DISTILLER.TYPE in ['RLD', 'SDD_RLD']:
        if args.same_t:
            cfg.RLD.ALPHA_T = cfg.RLD.T
        cfg.EXPERIMENT.TAG += ',at=' + str(cfg.RLD.ALPHA_T)
        cfg.EXPERIMENT.TAG += ',t=' + str(cfg.RLD.T)
        cfg.EXPERIMENT.TAG += ',alpha=' + str(cfg.RLD.ALPHA)
        cfg.EXPERIMENT.TAG += ',beta=' + str(cfg.RLD.BETA)

    if cfg.DISTILLER.TYPE in ['DKD', 'SDD_DKD']:
        cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.T)
        cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.ALPHA)
        cfg.EXPERIMENT.TAG += ',' + str(cfg.DKD.BETA)
    if args.M is None:
        cfg.M = '[1]'
    else:
        cfg.M = args.M

    if args.M != '[1]':
        print(f"Original M argument: {args.M}")
        cfg.M = args.M
        print(f"cfg.M value: {cfg.M}")
        cfg.EXPERIMENT.TAG += ',sdd'
        cfg.EXPERIMENT.TAG += ',' + str(cfg.M)

    cfg.freeze()
    main(cfg, args.resume, args.opts)