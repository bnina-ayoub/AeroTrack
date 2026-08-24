#!/bin/bash

# ==============================================================================
# Script d'automatisation des évaluations : Distance (IoU/NWD) x Early Exit
# ==============================================================================

# 1. Variables de configuration (À ajuster selon ton projet)
EXP_FILE="exps/aerotrack_proposed.py"
CKPT_FILE="early_exit_weights.pth" # <-- Remplace ceci par le chemin exact de tes poids

echo "========================================================================"
echo "🚀 Lancement des 4 expériences d'évaluation automatisées"
echo "========================================================================"

# 2. Définition des paramètres à tester
METRICS=("iou" "nwd")
EARLY_EXITS=("" "--early_exit") # Une chaîne vide pour False, le flag pour True

# 3. Boucles imbriquées pour croiser les paramètres
for METRIC in "${METRICS[@]}"; do
    for EE_FLAG in "${EARLY_EXITS[@]}"; do
        
        # Déterminer le nom du mode pour un affichage propre dans le terminal
        if [ -z "$EE_FLAG" ]; then
            MODE="Baseline (Early Exit: DESACTIVE)"
        else
            MODE="Proposé (Early Exit: ACTIVE)"
        fi

        echo ""
        echo "------------------------------------------------------------------------"
        echo "⏳ DÉMARRAGE : Distance = $METRIC | Mode = $MODE"
        echo "💻 Commande exécutée : sudo ~/AeroTrack/venv/bin/python tools/track.py --fp16 --fuse --trt -d 1 -b 1 -f $EXP_FILE --distance $METRIC $EE_FLAG --monitor_energy data_num_workers 0 "
        echo "------------------------------------------------------------------------"
        echo ""
        
        # 4. Exécution réelle de la commande Python
        sudo ~/AeroTrack/venv/bin/python tools/track.py --fp16 --fuse --trt -d 1 -b 1 -f $EXP_FILE --distance $METRIC $EE_FLAG --monitor_energy data_num_workers 0
        
        echo ""
        echo "✅ TERMINÉ : Distance = $METRIC | Mode = $MODE"
        echo "------------------------------------------------------------------------"
        
    done
done

echo ""
echo "========================================================================"
echo "🎉 Toutes les évaluations ont été exécutées avec succès !"
echo "========================================================================"
