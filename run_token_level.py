"""run_token_level.py - Zustandsdekodierungs-Experiment (Kapitel 3.1 / 4).

Direkt aufrufbar, kein Argumentparser:

    from run_token_level import run_token_level
    run_token_level()

Alle Parameter stehen zentral in params.py. Pro Durchlauf wird ein
frisches, zufälliges, dünn besetztes, alpha-gemischtes HMM erzeugt (siehe
rescrf.util.create_random_HMM_order_mix), auf dessen Stichprobe die
Kandidatenmodelle trainiert und zusätzlich das datengenerierende HMM
selbst ausgewertet werden (keine Training, siehe evaluate_true_model).

Absturzsicherheit: Jedes einzelne Ergebnis wird SOFORT nach Abschluss
unter <TOKENLEVEL_RESULTS_PATH>/partial_results/ gespeichert. Nach einem
Abbruch setzt ein erneuter Aufruf automatisch dort fort, wo er unterbrochen
wurde (siehe rescrf.training.cached_run). Diagramme lassen sich jederzeit,
auch aus einem unvollständigen Lauf, mit regenerate_plots.py neu erzeugen.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # unterdrückt harmlose TF/XLA-INFO/WARNING-C++-Logs

import tensorflow as tf

from rescrf.data import create_random_data, sample_from_hmm
from rescrf.models import get_experiments
from rescrf.training import (
    run_single_training, evaluate_true_model, cached_run,
    all_cached, load_cached, print_summary,
)
from rescrf.plotting import generate_all_plots, save_raw_results, plot_alpha_sweep, generate_aggregate_report, group_all_token_level
from rescrf.progress import make_load_bar, tick
import params as P


def _dataset_from_arrays(states, emissions, batch_size, num_symbols):
    """One-Hot-kodiertes tf.data.Dataset direkt aus im Speicher
    vorliegenden Tensoren (kein Umweg über die Festplatte - bei ~1000
    Durchläufen mit je neuem Datensatz würde Zwischenspeichern auf Disk
    keinen Zweck erfüllen)."""
    x = tf.one_hot(tf.cast(emissions, tf.int32), depth=num_symbols)
    df = tf.data.Dataset.from_tensor_slices((x, states))
    return df.repeat().shuffle(100).batch(batch_size)


def run_token_level():
    experiments = get_experiments("full" if P.ALL_MODELS else "base")
    expected_pairs = [("true_generator", 0.0)] + [
        (name, lr) for name in experiments for lr in P.LEARNING_RATES
    ]
    results_dir = P.TOKENLEVEL_RESULTS_PATH / "partial_results"

    all_results = []
    load_bar = make_load_bar(P.TOKENLEVEL_NUM_HMMS, f"Zustandsdekodierung ({P.TOKENLEVEL_NUM_HMMS} HMMs)")

    for i in range(P.TOKENLEVEL_NUM_HMMS):
        seed = P.TOKENLEVEL_HMM_SEED_START + i
        states, emissions, hmm, alpha_used = create_random_data(
            N=P.TOKENLEVEL_BATCH_SIZE, T=P.TOKENLEVEL_T,
            K=P.TOKENLEVEL_HMM_K, out_degree=P.TOKENLEVEL_HMM_OUT_DEGREE, M=P.TOKENLEVEL_HMM_M,
            alpha=None, alpha_range=P.TOKENLEVEL_HMM_ALPHA_RANGE, seed=seed, return_model=True,
        )
        num_symbols = P.TOKENLEVEL_HMM_M
        # M im Namen, damit ein Alphabet-Wechsel nie fälschlich mit
        # partial_results eines früheren Laufs mit abweichendem M kollidiert.
        hierarchy_config = f"hmm{i:04d}_seed{seed}_alpha{alpha_used:.3f}_M{num_symbols}"

        print(f"\n[{i + 1}/{P.TOKENLEVEL_NUM_HMMS}] {hierarchy_config}")

        if all_cached(results_dir, hierarchy_config, expected_pairs):
            print("  bereits vollständig vorhanden - überspringe.")
            all_results.extend(load_cached(results_dir, hierarchy_config, expected_pairs))
            tick(load_bar)
            continue

        dataset = _dataset_from_arrays(states, emissions, P.TOKENLEVEL_BATCH_SIZE, num_symbols)

        # Auswertungsset (Abschnitt 4.4): unabhängig vom Training aus
        # DEMSELBEN hmm gezogene Sequenzen, weder für das Training der
        # Kandidatenmodelle noch für deren Konvergenz-/Auswahlkriterien
        # verwendet. Sowohl die trainierten Modelle als auch die
        # Referenzauswertung (evaluate_true_model) werden am Ende darauf
        # ausgewertet ("eval_loss"/"eval_accuracy"), damit der Vergleich
        # fair ist - siehe run_single_training-Docstring.
        eval_states, eval_emissions = sample_from_hmm(hmm, P.TOKENLEVEL_EVAL_SIZE, P.TOKENLEVEL_T)
        eval_dataset = _dataset_from_arrays(eval_states, eval_emissions, P.TOKENLEVEL_EVAL_SIZE, num_symbols)

        # Tatsächliche Zustandsanzahl (K^2) aus den Daten selbst ableiten -
        # MODELS-Konfigurationen enthalten bewusst kein festes "output"
        # mehr (siehe models.py), da es je Durchlauf von K abhängt.
        num_states = int(tf.reduce_max(states).numpy()) + 1

        true_result = cached_run(
            results_dir, hierarchy_config, "true_generator", 0.0,
            lambda: evaluate_true_model(
                hmm=hmm, dataset=eval_dataset, T=P.TOKENLEVEL_T, steps=1,
                hierarchy_config=hierarchy_config, results_dir=results_dir, num_symbols=num_symbols,
            ),
        )
        all_results.append(true_result)

        for exp_name, base_config in experiments.items():
            model_config = {**base_config, "output": num_states}
            for lr in P.LEARNING_RATES:
                result = cached_run(
                    results_dir, hierarchy_config, exp_name, lr,
                    lambda exp_name=exp_name, model_config=model_config, lr=lr: run_single_training(
                        exp_name=exp_name, model_config=model_config, learning_rate=lr,
                        T=P.TOKENLEVEL_T, weight_decay=P.WEIGHT_DECAY, dataset=dataset,
                        eval_dataset=eval_dataset,
                        save_checkpoints=P.SAVE_CHECKPOINTS,
                        convergence_threshold=P.CONVERGENCE_THRESHOLD,
                        convergence_patience=P.CONVERGENCE_PATIENCE,
                        best_value_patience=P.BEST_VALUE_PATIENCE,
                        max_epochs=P.MAX_EPOCHS, steps_per_epoch=P.STEPS_PER_EPOCH,
                        task="token_level", hierarchy_config=hierarchy_config,
                        results_dir=results_dir, num_symbols=num_symbols,
                    ),
                )
                all_results.append(result)

        tick(load_bar)

    print_summary(all_results)
    save_raw_results(all_results, P.TOKENLEVEL_RESULTS_PATH / "results.json")

    ok_results = [r for r in all_results if r.error is None]
    generate_all_plots(ok_results, P.TOKENLEVEL_RESULTS_PATH, metrics=("loss", "accuracy"))
    for metric in ("loss", "accuracy"):
        plot_alpha_sweep(
            ok_results, exp_names=P.TOKENLEVEL_ALPHA_SWEEP_MODELS,
            out_path=P.TOKENLEVEL_RESULTS_PATH / f"alpha_sweep_{metric}.png", metric=metric,
        )
    # Derselbe Sweep auf dem Auswertungsset ("eval_loss"/"eval_accuracy")
    # - der eigentlich faire Vergleich zwischen Referenz und trainierten
    # Modellen, siehe run_single_training-Docstring.
    for metric in ("eval_loss", "eval_accuracy"):
        plot_alpha_sweep(
            ok_results, exp_names=P.TOKENLEVEL_ALPHA_SWEEP_MODELS,
            out_path=P.TOKENLEVEL_RESULTS_PATH / f"alpha_sweep_{metric}.png", metric=metric,
        )

    # Mittelwert/Std. über alle TOKENLEVEL_NUM_HMMS Durchläufe hinweg
    # (K/out_degree/M sind über alle Durchläufe konstant, siehe
    # group_all_token_level) - Abschnitt 4.4 "Metriken und Wiederholungen".
    # EINMAL auf den Trainingsmetriken (Diagnose: wie gut wird die feste
    # Trainingsstichprobe gefittet) und EINMAL auf dem unabhängigen
    # Auswertungsset (der für den Modellvergleich maßgebliche Bericht,
    # inkl. fairer Referenzauswertung).
    generate_aggregate_report(ok_results, P.TOKENLEVEL_RESULTS_PATH, group_fn=group_all_token_level)
    generate_aggregate_report(
        ok_results, P.TOKENLEVEL_RESULTS_PATH / "eval", group_fn=group_all_token_level,
        metrics=("eval_loss", "eval_accuracy"),
    )

    print(f"\nErgebnisse: {P.TOKENLEVEL_RESULTS_PATH.resolve()}")
    return all_results


if __name__ == "__main__":
    run_token_level()
