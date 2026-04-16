"""
Génération des graphiques de métriques d'entraînement.

Pour chaque environnement, produit un fichier image par métrique montrant
la courbe moyenne +- écart-type de chaque algorithme.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ===========================================================================
# CONFIGURATION - ajouter ou retirer des dossiers selon les besoins.
# Chaque dossier représente un algorithme entraîné sur un environnement donné.
# Convention de nommage attendue : "{algo}_{environnement}"
# ===========================================================================
DOSSIERS = [
    "ppo_10x10_custom_maxsteps60",
    "ppo_10x10_custom_maxsteps100",
    "ppo_10x10_custom_maxsteps150",
    "ppo_16x16_random_maxsteps60",
    "ppo_16x16_random_maxsteps100",
    "ppo_16x16_random_maxsteps150"
]

#Nom du paramètre que l'on a fait varier
#param_name = "maxsteps"

#DOSSIERS = [
#    "ppo_10x10_custom_lr00005",
#    "ppo_10x10_custom_lr0003",
#    "ppo_10x10_custom_lr001",
#]

#Nom du paramètre que l'on a fait varier
#param_name = "lr"

# Dossier de sortie pour les figures
FIGURES_DIR = "Figures"

# Algorithmes connus - utilisés pour extraire le nom d'environnement depuis le nom du dossier
ALGOS_CONNUS = {
    "q_learning": "Q-learning", 
    "ppo": "PPO"
}

# Environnements connus - utilisés pour l'affichage dans les titres
ENVS_CONNUS = {
    "8x8_default": "8x8 par défaut",
    "10x10_custom": "10x10 personnalisé",
    "16x16_random": "16x16 aléatoire",
}

# Métriques à tracer.
# Chaque entrée : (label_affichage, clé_averaged_metrics, clé_callback, use_x_raw)
#   - label_affichage    : titre de l'axe Y et du graphique
#   - clé_averaged_metrics : clé dans averaged_metrics.npz (valeur moyenne)
#   - clé_callback       : clé dans callback_*.npz (valeurs par seed pour le std)
#   - use_x_raw          : True -> utiliser x_raw sans lissage (ex: cum_falls)
#                          False -> utiliser x_ma avec moyenne mobile
METRIQUES = [
    ("Récompense moyenne",         "reward_mean",               "episode_rewards",               False),
    ("Longueur d'épisode (succès)","length_given_success_mean", "episode_lengths_given_success", False),
    ("Taux de succès",             "success_rate",              "episode_successes",             False),
    ("Taux de chutes",             "fall_rate",                 "episode_falls",                 False),
    ("Risque moyen",               "risk_mean",                 "episode_risks",                 False),
]


# ==================== Fonctions utilitaires ====================

def extraire_infos(nom_dossier: str):
    """
    Extrait algo, environnement et param_value

    Exemple :
    ppo_10x10_custom_maxsteps100 ->
    ("ppo", "10x10_custom", "100")
    """
    parties = nom_dossier.split("_")

    algo = parties[0]
    param_value = None

    # détecter le paramètre
    for p in parties:
        if p.startswith(param_name):
            param_value = p.replace(param_name, "")
            break

    if param_value is None:
        raise ValueError(f"{param_name} non trouvé dans {nom_dossier}")

    # reconstruire env sans le param
    env = "_".join([p for p in parties[1:] if not p.startswith(param_name)])

    return algo, env, param_value

def extraire_resume_dossier_callbacks(dossier: str):
    """
    Calcule la moyenne des métriques pendant tout l'entraînement en utilisant les callbacks (valeurs brutes).

    retourne un dictionnaire contenant :
            - env : nom de l'environnement
            - param_name : valeur du paramètre (ex: maxsteps)
            - une clé par métrique avec la moyenne globale correspondante
    """
    algo, env, param_value = extraire_infos(dossier)
    _, callbacks = charger_dossier(dossier)

    resume = {
        "env": env,
        param_name: int(param_value),
    }

    # Pour chaque métrique définie dans METRIQUES
    for label, _, cle_callback, _ in METRIQUES:

        # Stocker toutes les valeurs de tous les seeds
        toutes_valeurs = []

        for cb in callbacks:
            # Extraire les données brutes (par épisode)
            data = cb[cle_callback].astype(float)

            toutes_valeurs.append(data)

        # Concaténer toutes les séries (tous les épisodes, tous les seeds)
        toutes_valeurs = np.concatenate(toutes_valeurs)

        # Moyenne globale en ignorant les NaN
        moyenne = np.nanmean(toutes_valeurs)

        resume[label] = moyenne

    return resume

def moving_average(x: np.ndarray, fenetre: int) -> np.ndarray:
    """
    Calcule la moyenne mobile en ignorant les NaN (comportement identique à utils.py).

    Retourne un tableau de longueur len(x) - fenetre + 1.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < fenetre:
        return x
    kernel = np.ones(fenetre)
    somme = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="valid")
    nb_valides = np.convolve(~np.isnan(x), kernel, mode="valid")
    avec_nan = somme / nb_valides
    avec_nan[nb_valides == 0] = np.nan
    return avec_nan


def charger_dossier(dossier: str) -> tuple[np.lib.npyio.NpzFile, list]:
    """
    Charge averaged_metrics.npz et tous les fichiers callback_*.npz du dossier.

    Retourne (averaged_metrics, liste_de_callbacks).
    """
    chemin_avg = os.path.join(dossier, "averaged_metrics.npz")
    if not os.path.exists(chemin_avg):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_avg}")
    averaged = np.load(chemin_avg)

    # Charger tous les fichiers callback dans l'ordre croissant (par seed)
    fichiers_cb = sorted(glob.glob(os.path.join(dossier, "callback_*.npz")))
    if not fichiers_cb:
        raise FileNotFoundError(f"Aucun fichier callback_*.npz trouvé dans : {dossier}")
    callbacks = [np.load(f) for f in fichiers_cb]

    return averaged, callbacks


def calculer_std_seeds(
    callbacks: list,
    cle: str,
    fenetre: int,
    use_x_raw: bool,
) -> np.ndarray:
    """
    Calcule l'écart-type entre seeds pour une métrique donnée.

    Pour les métriques lissées : applique la même moyenne mobile que celle
    utilisée pour produire averaged_metrics, puis calcule le std entre seeds.
    Pour cum_falls (use_x_raw=True) : utilise les valeurs brutes directement.
    """
    series = []
    for cb in callbacks:
        donnees = cb[cle].astype(float)
        if use_x_raw:
            series.append(donnees)
        else:
            series.append(moving_average(donnees, fenetre))

    # Tronquer toutes les séries à la longueur minimale pour pouvoir empiler
    longueur_min = min(len(s) for s in series)
    series_tronquees = np.array([s[:longueur_min] for s in series])

    # nanstd pour gérer les NaN éventuels (ex: length_given_success)
    return np.nanstd(series_tronquees, axis=0)


# ==================== Génération des graphiques ====================

def tracer_metrique(
    ax,
    label: str,
    cle_averaged: str,
    cle_callback: str,
    use_x_raw: bool,
    dossiers_env: list[str],
) -> None:
    """
    Trace une métrique sur l'axe fourni avec tous les algorithmes superposés.

    Chaque algorithme est représenté par sa courbe moyenne +- écart-type entre seeds.
    """
    couleurs = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    dossiers_env = sorted(
        dossiers_env,
        key=lambda d: int(extraire_infos(d)[2])  # tri par param_value
    )

    for idx, dossier in enumerate(dossiers_env):
        algo, _, param_value = extraire_infos(dossier)
        averaged, callbacks = charger_dossier(dossier)

        # Sélectionner l'axe X et la série moyenne selon le type de métrique
        if use_x_raw:
            x = averaged["x_raw"]
            y_moy = averaged[cle_averaged]
        else:
            x = averaged["x_ma"]
            y_moy = averaged[cle_averaged]

        # Déduire la fenêtre de lissage depuis x_ma : x_ma[0] == fenetre
        fenetre = int(averaged["x_ma"][0])

        # Calculer l'écart-type entre seeds
        std = calculer_std_seeds(callbacks, cle_callback, fenetre, use_x_raw)

        # Tronquer à la longueur commune (les séries doivent être alignées)
        n = min(len(x), len(y_moy), len(std))
        x, y_moy, std = x[:n], y_moy[:n], std[:n]

        couleur = couleurs[idx % len(couleurs)]

        label = f"{ALGOS_CONNUS[algo]} - {param_name} = {param_value}"
        ax.plot(x, y_moy, label=label, color=couleur, linewidth=1.5)
        ax.fill_between(x, y_moy - std, y_moy + std, alpha=0.2, color=couleur)

    ax.set_xlabel("Épisode ($\\times 10^3$)")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Formater l'axe des x en milliers
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}'))


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Regrouper les dossiers par environnement
    groupes_env: dict[str, list[str]] = {}
    for dossier in DOSSIERS:
        if not os.path.isdir(dossier):
            print(f"[AVERTISSEMENT] Dossier introuvable, ignoré : {dossier}")
            continue
        _, env, param_value = extraire_infos(dossier)
        if env not in ENVS_CONNUS:
            print(f"[AVERTISSEMENT] Environnement inconnu dans le nom du dossier, ignoré : {dossier}")
            continue
        groupes_env.setdefault(env, []).append(dossier)

    if not groupes_env:
        print("Aucun dossier valide trouvé. Vérifiez la liste DOSSIERS.")
        return

    # Créer une figure avec tous les métriques et environnements
    num_metrics = len(METRIQUES)
    num_envs = len(groupes_env)
    height = max(4 + num_metrics * 1.5, 10)  # Hauteur minimale de 10, ajouter 1.5 par métrique
    fig, axes = plt.subplots(num_metrics, num_envs, figsize=(4 * num_envs, height), sharex='col', sharey='row')
    
    # Gérer le cas où il y a seulement 1 métrique ou 1 environnement (les axes pourraient ne pas être 2D)
    if num_metrics == 1 or num_envs == 1:
        axes = axes.reshape(num_metrics, num_envs) if not isinstance(axes, np.ndarray) or axes.ndim == 1 else axes

    print(f"Génération d'une figure avec {num_metrics} métriques et {num_envs} environnement(s)")

    # Itérer sur les métriques (lignes) et environnements (colonnes)
    for metric_idx, (label, cle_averaged, cle_callback, use_x_raw) in enumerate(METRIQUES):
        for env_idx, (nom_env, dossiers_env) in enumerate(groupes_env.items()):
            try:
                ax = axes[metric_idx, env_idx]
                tracer_metrique(
                    ax, label, cle_averaged, cle_callback, use_x_raw, dossiers_env
                )
                # Définir le titre de colonne uniquement pour la première ligne
                if metric_idx == 0:
                    ax.set_title(f"{ENVS_CONNUS[nom_env]}", fontsize=12, fontweight="bold")
                else:
                    ax.set_title("")
                
                # Définir l'étiquette de ligne uniquement pour la première colonne
                if env_idx == 0:
                    ax.set_ylabel(label, fontsize=11, fontweight="bold")
                else:
                    ax.set_ylabel("")
                
                print(f"  [{metric_idx+1}/{num_metrics}][{env_idx+1}/{num_envs}] {label} - {nom_env} - {param_name} - {param_value} - OK")
            except Exception as e:
                print(f"  [ERREUR] {label} - {nom_env} : {e}")

    fig.tight_layout()
    nom_fichier = f"grid_search_{param_name}.png"
    chemin_sortie = os.path.join(FIGURES_DIR, nom_fichier)
    fig.savefig(chemin_sortie, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSauvegardé : {chemin_sortie}")

    # ==================== TABLEAU RÉSUMÉ ====================
    
    lignes = []

    for dossier in DOSSIERS:
        if not os.path.isdir(dossier):
            continue
        try:
            lignes.append(extraire_resume_dossier_callbacks(dossier))
        except Exception as e:
            print(f"[ERREUR TABLEAU] {dossier}: {e}")

    df = pd.DataFrame(lignes)

    # trier pour que ce soit lisible
    df = df.sort_values(by=["env", param_name])

    # sauvegarde CSV
    chemin_csv = os.path.join(FIGURES_DIR, f"averaged_training_metrics_grid_search_{param_name}.csv")
    df.to_csv(chemin_csv, index=False)

    print("\nTableau résumé :")
    print(df)
    print(f"\nSauvegardé : {chemin_csv}")


if __name__ == "__main__":
    main()
