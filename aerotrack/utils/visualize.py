#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis", "plot_tracking"]

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    # Tactical HUD parameters for static detections
    tactical_color = (0, 255, 0) # Neon Green (BGR)
    
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        if score < conf:
            continue
            
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Formatting text
        cls_name = class_names[int(cls_ids[i])] if class_names else "OBJ"
        text = f'TGT_ACQ [{cls_name.upper()}]: {score * 100:.1f}%'
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        
        # Draw thin bounding box
        cv2.rectangle(img, (x0, y0), (x1, y1), tactical_color, 1)

        # Draw dark background for HUD text
        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 4),
            (x0 + txt_size[0] + 2, y0),
            (0, 0, 0),
            -1
        )
        # Put tactical green text
        cv2.putText(img, text, (x0 + 1, y0 - 3), font, 0.4, tactical_color, thickness=1)

    return img


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # Military HUD UI Settings
    tactical_color = (0, 255, 0)  # Neon Green
    alert_color = (0, 0, 255)     # Red for critical system text
    text_thickness = 1
    bracket_thickness = 2
    
    # Global HUD Telemetry (Top Left)
    hud_text = f'SYS_FRAME: {frame_id:06d} | FPS: {fps:.1f} | TGTS_LOCKED: {len(tlwhs)}'
    cv2.putText(im, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tactical_color, thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        
        # Format ID as a military track number (e.g., TRK-005)
        id_text = f'TRK-{obj_id:03d}'
        if ids2 is not None:
            id_text += f' [{int(ids2[i])}]'

        # 1. Calculate center and bracket lengths
        cx, cy = int(x1 + w / 2), int(y1 + h / 2)
        corner_len = int(min(w, h) * 0.25)  # Brackets take up 25% of the shortest side

        # 2. Draw Corner Brackets (Targeting Reticle Style)
        # Top-Left
        cv2.line(im, (intbox[0], intbox[1]), (intbox[0] + corner_len, intbox[1]), tactical_color, bracket_thickness)
        cv2.line(im, (intbox[0], intbox[1]), (intbox[0], intbox[1] + corner_len), tactical_color, bracket_thickness)
        # Top-Right
        cv2.line(im, (intbox[2], intbox[1]), (intbox[2] - corner_len, intbox[1]), tactical_color, bracket_thickness)
        cv2.line(im, (intbox[2], intbox[1]), (intbox[2], intbox[1] + corner_len), tactical_color, bracket_thickness)
        # Bottom-Left
        cv2.line(im, (intbox[0], intbox[3]), (intbox[0] + corner_len, intbox[3]), tactical_color, bracket_thickness)
        cv2.line(im, (intbox[0], intbox[3]), (intbox[0], intbox[3] - corner_len), tactical_color, bracket_thickness)
        # Bottom-Right
        cv2.line(im, (intbox[2], intbox[3]), (intbox[2] - corner_len, intbox[3]), tactical_color, bracket_thickness)
        cv2.line(im, (intbox[2], intbox[3]), (intbox[2], intbox[3] - corner_len), tactical_color, bracket_thickness)

        # 3. Draw Faint Connecting Box
        cv2.rectangle(im, intbox[0:2], intbox[2:4], (0, 100, 0), 1)

        # 4. Draw Center Crosshair
        cv2.line(im, (cx - 4, cy), (cx + 4, cy), tactical_color, 1)
        cv2.line(im, (cx, cy - 4), (cx, cy + 4), tactical_color, 1)

        # 5. Draw ID Text with Black Background for Readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(id_text, font, 0.45, text_thickness)[0]
        
        cv2.rectangle(im, 
                      (intbox[0], intbox[1] - txt_size[1] - 6), 
                      (intbox[0] + txt_size[0] + 4, intbox[1]), 
                      (0, 0, 0), -1)
                      
        cv2.putText(im, id_text, (intbox[0] + 2, intbox[1] - 4), 
                    font, 0.45, tactical_color, thickness=text_thickness)

    return im