"""run_classification.py - Sequenzklassifikations-Experiment (Kapitel 3.2 / 4).

Direkt aufrufbar, kein Argumentparser:

    from run_classification import run_classification
    run_classification()

Alle Parameter stehen zentral in params.py. Für jede Kombination aus
Hierarchietiefe D, Mustern pro Ebene K, Basislänge L und Klassenzahl C
wird CLASS_NUM_RUNS-mal ein neuer Datensatz gezogen und die
Kandidatenmodelle darauf trainiert (siehe shmm_moduls.hierarchical_data).

Absturzsicherheit: wie run_token_level.py, siehe dort.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # unterdrückt harmlose TF/XLA-INFO/WARNING-C++-Logs

import itertools

import numpy as np
import tensorflow as tf

from shmm_moduls.hierarchical_data import create_hierarchical_data
from shmm_moduls.models import get_experiments
from shmm_moduls.training import run_single_training, cached_run, all_cached, load_cached, print_summary
from shmm_moduls.plotting import generate_all_plots, save_raw_results, generate_aggregate_report, group_by_classification_config
from shmm_moduls.progress import make_load_bar, tick
import params as P


def _configs() -> list[dict]:
    """Alle Kombinationen der Hierarchie-Parameter aus params.py, je mit
    einem eindeutigen Namen (siehe daten.tex, "Konkrete Parametrisierung")."""
    combos = itertools.product(
        P.CLASS_HIERARCHY_LEVELS, P.CLASS_PATTERNS_PER_LEVEL,
        P.CLASS_BASE_PATTERN_LENGTH, P.CLASS_NUM_CLASSES,
    )
    return [
        {"D": D, "K": K, "L": L, "classes": C, "name": f"D{D}_K{K}_L{L}_C{C}_M{P.CLASS_ALPHABET_SIZE}"}
        for D, K, L, C in combos
    ]


def _dataset(config: dict, seed: int):
    """Erzeugt Trainings- UND Auswertungsset: Letzteres verwendet
    DIESELBEN Klassen-Templates (`return_templates`/`templates`,
    hierarchical_data.create_hierarchical_data), aber unabhängig davon
    gezogenes Rauschen (`instance_seed`), sodass beide Mengen dieselbe
    Aufgabe, aber disjunkte konkrete Sequenzen enthalten (Abschnitt 4.4).
    """
    emissions, states, class_templates = create_hierarchical_data(
        N=P.CLASS_N_SAMPLES, L=config["L"], classes=config["classes"], K=config["K"], D=config["D"],
        alpha=P.CLASS_NOISE_ALPHA, head_noise=False, tail_noise=None, check_uniqueness=True,
        seed=seed, alphabet_size=P.CLASS_ALPHABET_SIZE, return_templates=True,
    )
    eval_emissions, eval_states = create_hierarchical_data(
        N=P.CLASS_EVAL_SIZE, L=config["L"], classes=config["classes"], K=config["K"], D=config["D"],
        alpha=P.CLASS_NOISE_ALPHA, head_noise=False, tail_noise=None, check_uniqueness=True,
        alphabet_size=P.CLASS_ALPHABET_SIZE, templates=class_templates,
        instance_seed=seed + P.CLASS_EVAL_SEED_OFFSET,
    )

    num_classes = states.shape[1]

    # Padding-Länge kann zwischen Trainings- und Auswertungsset leicht
    # abweichen (max_len hängt von den längsten instanziierten Sequenzen
    # der jeweiligen Stichprobe ab) - auf die GEMEINSAME maximale Länge
    # angleichen, damit Trainings- und Eval-Modell dieselbe Input-Shape T
    # verwenden. Wie in hierarchical_data._instantiate wird dabei mit
    # gleichverteilten Zufallssymbolen aufgefüllt (nicht mit Nullen), um
    # dem Modell keine künstliche, im Training nie gesehene
    # "Padding-Kategorie" zu präsentieren.
    T = max(emissions.shape[1], eval_emissions.shape[1])

    def _pad_to(arr: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
        if arr.shape[1] >= length:
            return arr
        pad = rng.integers(0, P.CLASS_ALPHABET_SIZE, size=(arr.shape[0], length - arr.shape[1]))
        return np.concatenate([arr, pad], axis=1)

    pad_rng = np.random.default_rng(seed)
    emissions = _pad_to(emissions, T, pad_rng)
    eval_emissions = _pad_to(eval_emissions, T, pad_rng)

    x = tf.one_hot(emissions, depth=P.CLASS_ALPHABET_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices((x, states)).shuffle(1000).repeat().batch(P.CLASS_BATCH_SIZE)

    eval_x = tf.one_hot(eval_emissions, depth=P.CLASS_ALPHABET_SIZE)
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_x, eval_states)).batch(P.CLASS_EVAL_SIZE)

    return dataset, eval_dataset, T, num_classes


def run_classification():
    experiments = get_experiments("full" if P.ALL_MODELS else "base_class")
    expected_pairs = [(name, lr) for name in experiments for lr in P.LEARNING_RATES]
    configs = _configs()
    results_dir = P.CLASS_RESULTS_PATH / "partial_results"

    total = len(configs) * P.CLASS_NUM_RUNS
    load_bar = make_load_bar(total, f"Sequenzklassifikation ({len(configs)} Konfig. x {P.CLASS_NUM_RUNS} Durchläufe)")

    all_results = []
    counter = 0
    for config in configs:
        for run_idx in range(P.CLASS_NUM_RUNS):
            counter += 1
            seed = P.CLASS_RUN_SEED_START + run_idx
            hierarchy_config = f"{config['name']}_run{run_idx:04d}"
            print(f"\n[{counter}/{total}] {hierarchy_config}")

            if all_cached(results_dir, hierarchy_config, expected_pairs):
                print("  bereits vollständig vorhanden - überspringe.")
                all_results.extend(load_cached(results_dir, hierarchy_config, expected_pairs))
                tick(load_bar)
                continue

            dataset, eval_dataset, T, num_classes = _dataset(config, seed)

            for exp_name, base_config in experiments.items():
                model_config = {**base_config, "output": num_classes}
                for lr in P.LEARNING_RATES:
                    result = cached_run(
                        results_dir, hierarchy_config, exp_name, lr,
                        lambda exp_name=exp_name, model_config=model_config, lr=lr, dataset=dataset,
                        eval_dataset=eval_dataset, T=T:
                        run_single_training(
                            exp_name=exp_name, model_config=model_config, learning_rate=lr,
                            T=T, weight_decay=P.WEIGHT_DECAY, dataset=dataset,
                            eval_dataset=eval_dataset,
                            save_checkpoints=P.SAVE_CHECKPOINTS,
                            convergence_threshold=P.CONVERGENCE_THRESHOLD,
                            convergence_patience=P.CONVERGENCE_PATIENCE,
                            best_value_patience=P.BEST_VALUE_PATIENCE,
                            max_epochs=P.MAX_EPOCHS, steps_per_epoch=P.STEPS_PER_EPOCH,
                            task="classification", hierarchy_config=hierarchy_config,
                            results_dir=results_dir, num_symbols=P.CLASS_ALPHABET_SIZE,
                        ),
                    )
                    all_results.append(result)

            tick(load_bar)

    print_summary(all_results)
    save_raw_results(all_results, P.CLASS_RESULTS_PATH / "results.json")

    ok_results = [r for r in all_results if r.error is None]
    generate_all_plots(ok_results, P.CLASS_RESULTS_PATH, metrics=("loss", "accuracy"))

    # Mittelwert/Std. über alle CLASS_NUM_RUNS Wiederholungen JEDER
    # Parameterkombination (D,K,L,C,M) getrennt, siehe
    # group_by_classification_config - Abschnitt 4.4. Zusätzlich zu den
    # Trainingsmetriken ("loss"/"accuracy") derselbe Bericht auf dem
    # unabhängigen Auswertungsset ("eval_loss"/"eval_accuracy") - der für
    # den Modellvergleich maßgebliche.
    generate_aggregate_report(ok_results, P.CLASS_RESULTS_PATH, group_fn=group_by_classification_config)
    generate_aggregate_report(
        ok_results, P.CLASS_RESULTS_PATH / "eval", group_fn=group_by_classification_config,
        metrics=("eval_loss", "eval_accuracy"),
    )

    print(f"\nErgebnisse: {P.CLASS_RESULTS_PATH.resolve()}")
    return all_results


if __name__ == "__main__":
    run_classification()
