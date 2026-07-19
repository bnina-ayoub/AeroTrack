#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import importlib
import os
import sys
import importlib.util

def get_exp_by_file(exp_file):
    try:
        # 1. Obtention des chemins absolus
        abs_exp_file = os.path.abspath(exp_file)
        exp_dir = os.path.dirname(abs_exp_file)
        exp_name = os.path.basename(abs_exp_file).split(".")[0]
        
        # 2. Ajout du dossier au sys.path pour permettre les imports internes 
        # (ex: pouvoir faire "from uavswarm_base import ...")
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        
        # 3. Chargement explicite et direct du fichier (Méthode robuste Python 3.5+)
        spec = importlib.util.spec_from_file_location(exp_name, abs_exp_file)
        if spec is None:
            raise ImportError(f"Spécification introuvable pour le fichier : {abs_exp_file}")
        
        current_exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(current_exp)
        
        # 4. Instanciation de la classe Exp
        exp = current_exp.Exp()
        
    except Exception as e:
        # Remontée de l'erreur avec le détail précis
        raise ImportError(f"Impossible de charger '{exp_file}'. Erreur d'origine : {e}")
        
    return exp

def get_exp_by_name(exp_name):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)
