"""regenerate_plots.py - Erstellt/aktualisiert Diagramme (inkl. Mittelwert/
Std. über alle Wiederholungen) aus zwischengespeicherten Trainings-
ergebnissen, unabhängig vom Training selbst (liest nur die JSON-Dateien
unter <results_dir>/partial_results/).

Nützlich um Diagramme zu sehen, während ein Training noch läuft, oder um
sie aus einem abgebrochenen/abgestürzten Lauf neu zu erzeugen, ohne neu zu
trainieren.

Direkt aufrufbar, kein Argumentparser:

    from regenerate_plots import regenerate_plots
    from rescrf.plotting import group_all_token_level, group_by_classification_config
    regenerate_plots(params.TOKENLEVEL_RESULTS_PATH, alpha_sweep=params.TOKENLEVEL_ALPHA_SWEEP_MODELS,
                      group_fn=group_all_token_level)
    regenerate_plots(params.CLASS_RESULTS_PATH, group_fn=group_by_classification_config)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # unterdrückt harmlose TF/XLA-INFO/WARNING-C++-Logs

from pathlib import Path

from shmm_moduls.training import load_all_results
from shmm_moduls.plotting import (
    generate_all_plots, save_raw_results, plot_alpha_sweep,
    generate_aggregate_report, group_by_classification_config,
)


def regenerate_plots(
    results_dir: Path, metrics: tuple[str, ...] = ("loss", "accuracy"),
    eval_metrics: tuple[str, ...] | None = ("eval_loss", "eval_accuracy"),
    alpha_sweep: list[str] | None = None, group_fn=group_by_classification_config,
    timing_baseline: str = "median",
) -> None:
    """
    `metrics` erzeugt wie bisher den Bericht auf den WÄHREND DES TRAININGS
    beobachteten Metriken (Trainingsstichprobe der besten Epoche) direkt
    unter `results_dir`. `eval_metrics` erzeugt zusätzlich, in einem
    eigenen Unterordner `results_dir/eval`, denselben Bericht auf dem in
    Abschnitt 3.3/4.4 beschriebenen, unabhängigen Auswertungsset - das
    ist die für den eigentlichen Modellvergleich maßgebliche Variante
    (siehe run_token_level.py/run_classification.py, die beide Berichte
    erzeugen). Ergebnisse aus älteren, vor Einführung des Auswertungssets
    zwischengespeicherten Läufen enthalten `eval_loss`/`eval_accuracy`
    nicht; in diesem Fall wird der eval-Bericht übersprungen, statt einen
    leeren/falschen Bericht zu erzeugen. Um nur den alten Trainingsbericht
    zu erzeugen, `eval_metrics=None` übergeben.
    """
    results_dir = Path(results_dir)
    partial_dir = results_dir / "partial_results"
    if not partial_dir.exists():
        print(f"Kein Verzeichnis mit Zwischenergebnissen gefunden: {partial_dir}")
        return

    results = load_all_results(partial_dir)
    if not results:
        print(f"Keine gespeicherten Ergebnisse in {partial_dir} gefunden.")
        return

    ok = [r for r in results if r.error is None]
    print(f"✓ {len(results)} Ergebnisse gefunden ({len(ok)} erfolgreich)")

    save_raw_results(results, results_dir / "results.json")
    if not ok:
        return

    generate_all_plots(ok, results_dir, metrics=metrics)

    if alpha_sweep:
        for metric in metrics:
            plot_alpha_sweep(ok, exp_names=alpha_sweep, out_path=results_dir / f"alpha_sweep_{metric}.png", metric=metric)

    generate_aggregate_report(ok, results_dir, group_fn=group_fn, metrics=metrics, timing_baseline=timing_baseline)

    if eval_metrics:
        has_eval = any(m in r.history for r in ok for m in eval_metrics)
        if has_eval:
            generate_aggregate_report(
                ok, results_dir / "eval", group_fn=group_fn, metrics=eval_metrics, timing_baseline=timing_baseline,
            )
            print(f"  ✓ Auswertungsset-Bericht zusätzlich unter: {(results_dir / 'eval').resolve()}")
        else:
            print("  (kein Auswertungsset in den zwischengespeicherten Ergebnissen gefunden - "
                  "eval-Bericht übersprungen; ältere Läufe ohne eval_loss/eval_accuracy?)")

    print(f"\n✓ Diagramme aktualisiert unter: {results_dir.resolve()}")


if __name__ == "__main__":
    import params as P
    from shmm_moduls.plotting import group_all_token_level

    regenerate_plots(P.TOKENLEVEL_RESULTS_PATH, alpha_sweep=P.TOKENLEVEL_ALPHA_SWEEP_MODELS, group_fn=group_all_token_level)
    regenerate_plots(P.CLASS_RESULTS_PATH, group_fn=group_by_classification_config)