"""Compact registry for the final Chapter-5 figures and tables."""

from __future__ import annotations

from typing import Any

CANONICAL_ARTIFACT_IDS = (
    "ch5_f1_paired_plr_fraud_tradeoff",
    "ch5_f2_budget_policy_profile",
    "ch5_f3_within_model_depth_k50",
    "ch5_f4_global_roc_pr_k50",
    "ch5_f5_hard_impact_profile_k50",
    "ch5_f6_replacement_case_map_k50",
    "app_f1_seed_budget_delta_heatmap",
    "app_f2_exact_tie_bound_intervals",
    "app_f3_candidate_pool_ceiling_utilization",
    "ch5_t1_central_topk_results",
    "ch5_t2_seedwise_k50_diagnostic",
    "ch5_t3_replacement_boundary_k50",
    "ch5_t4_high_amount_legit_k50",
    "app_t1_hard_impact_exact_values",
    "app_t2_global_metrics_by_budget",
    "app_t3_exact_tie_bounds",
    "app_t4_candidate_pool_coverage",
    "app_t5_seedwise_central_results",
)
ENGINEERING_ARTIFACT_IDS = (
    "engineering_f1_seed_budget_delta_heatmap",
    "engineering_t1_central_topk_summary",
)
ENGINEERING_COMPARABILITY_BOUNDARY = (
    "not comparable with canonical empirical results"
)


def _record(
    artifact_id: str,
    title: str,
    role: str,
    input_data_file: str,
    question: str,
    aggregation: str,
    caption: str,
    prohibited: str,
    *,
    additional: tuple[str, ...] = (),
    **extra: Any,
) -> dict[str, object]:
    row: dict[str, object] = {
        "artifact_id": artifact_id,
        "title": title,
        "role": role,
        "input_data_file": input_data_file,
        "empirical_question_answered": question,
        "aggregation_basis": aggregation,
        "caption_note": caption,
        "prohibited_interpretation": prohibited,
    }
    if additional:
        row["additional_input_data_files"] = list(additional)
    row.update(extra)
    return row


def build_selection_registry() -> dict[str, object]:
    """Return the interpretation-safe targeted R7B artifact selection."""

    budget_note = (
        "Trainierte Punkte stammen aus separaten budgetkonditionierten Modellen; "
        "die Linien verbinden ein geordnetes Policyprofil und sind keine "
        "Präfixauswertung eines einzelnen Rankers."
    )
    artifacts = [
        _record(
            "ch5_f1_paired_plr_fraud_tradeoff",
            "Gepaarter PLR–Fraud-Trade-off",
            "main-text",
            "data/figures/ch5_tradeoff_seedwise.csv",
            "Wie verändern die Vergleichspfade PLR@k und Fraud@k seedweise gegenüber BCE?",
            "Seed-gepaarte Differenzen; arithmetisches Mittel über n=5 Seeds",
            "Offene Rauten markieren Pfadmittel; einzelne Seeds erhalten keine Sonderannotation.",
            "Keine Signifikanz-, Kausalitäts-, Realverlust- oder Universalüberlegenheitsaussage.",
            additional=("data/figures/ch5_tradeoff_summary.csv",),
        ),
        _record(
            "ch5_f2_budget_policy_profile",
            "Diskretes Budget-Policyprofil",
            "main-text",
            "data/figures/ch5_budget_policy_summary.csv",
            "Wie hängen PLR@k und Fraud@k-Differenz von den sieben Budgetpolitiken ab?",
            "Arithmetisches Seedmittel je diskretem Budget und Pfad",
            budget_note,
            "Die Verbindungslinien sind weder Interpolation noch Präfixkurve eines globalen Rankers.",
            additional=("data/figures/ch5_budget_policy_seedwise.csv",),
        ),
        _record(
            "ch5_f3_within_model_depth_k50",
            "Rangtiefe innerhalb des fixen Budgetmodells k=50",
            "main-text",
            "data/figures/ch5_depth_summary.csv",
            "Wie entwickeln sich kumulatives PLR und Fraud-Zahl bis Rang 100 innerhalb des fixen Budgetmodells k=50?",
            "Kumulative Präfixe derselben eingefrorenen k=50-Vollordnung je Seed und Pfad; danach arithmetisches Seedmittel",
            "k=50 ist das Modellselektions-/Trainingsbudget; r ist die diagnostische Auslesetiefe innerhalb der festen abgeschlossenen Rangfolge dieses Modells.",
            "Keine Subtraktion oder Gleichsetzung verschiedener budgetkonditionierter Modelle; r ist kein neues Trainingsbudget.",
            additional=("data/figures/ch5_depth_seedwise.csv",),
        ),
        _record(
            "ch5_f4_global_roc_pr_k50",
            "Globale ROC- und Precision-Recall-Ordnung bei k=50",
            "main-text",
            "data/figures/ch5_global_pool_curves_summary.csv",
            "Wie unterscheiden sich die vier Pfade in ROC- und Precision-Recall-Ordnung im vollständigen Testbestand?",
            "Seedweise Full-Order-ROC/PR, gemeinsames 1001-Punkte-Gitter und arithmetisches Mittel über fünf Seeds",
            "Vollständige Testordnung bei k=50. Mittlere ROC-AUC/AP und BCE-Brier stehen in den Caption-Metadaten beziehungsweise in app_t2_global_metrics_by_budget; Brier ist ausschließlich für BCE definiert.",
            "Ordinale Rankerscores sind keine Fraud-Wahrscheinlichkeiten; keine Kandidatenpool- oder Kalibrierungsinterpretation.",
            additional=("data/figures/ch5_global_metrics_summary.csv",),
            caption_metric_data_file="data/figures/ch5_global_metrics_summary.csv",
        ),
        _record(
            "ch5_f5_hard_impact_profile_k50",
            "High-Amount-Fraud-Profil bei k=50",
            "optional-main-text",
            "data/figures/ch5_hard_impact_seedwise.csv",
            "Wie verhalten sich q90-Capture, Recovery von BCE-Fehlschlägen und Amount-nDCG@50?",
            "Fünf Outer-Split-Werte und arithmetisches Mittel je Pfad",
            "Baseline Miss Recovery ist für BCE nicht anwendbar und bleibt fehlend.",
            "Amount ist ein Proxy; keine Aussage über realisierten oder verhinderten finanziellen Verlust.",
            additional=("data/figures/ch5_hard_impact_summary.csv",),
        ),
        _record(
            "ch5_f6_replacement_case_map_k50",
            "Zusammensetzung der Top-50-Replacement-Ereignisse",
            "main-text",
            "data/figures/ch5_replacement_events.csv",
            "Welche Fälle fügt Amount-Gain hinzu beziehungsweise entfernt es gegenüber BCE?",
            "Seed-spezifische, per stabilem row_index ausgerichtete Replacement-Events",
            "Beobachtungen sind seed-spezifische Replacement-Events; dieselbe Transaktion kann in mehreren Outer-Splits auftreten. Amount ist ein Proxy. Die Abbildung beschreibt die beobachtete Selektionszusammensetzung, keine gelernte Entscheidungsgrenze und keine kausale Beziehung.",
            "Keine kausale Trend-, Entscheidungsgrenzen- oder Realverlustinterpretation; Ereigniszahlen sind keine Anzahl eindeutiger Transaktionen.",
        ),
        _record(
            "app_f1_seed_budget_delta_heatmap",
            "Seed-Budget-Matrix der Amount-Gain-Differenzen",
            "appendix",
            "data/figures/app_seed_budget_delta_heatmap.csv",
            "Wie verteilen sich Amount-Gain-Differenzen über alle fünf Seeds und sieben Budgets?",
            "Unaggregierte vollständige 5×7-Matrix",
            "Metriken besitzen getrennte, bei null zentrierte Farbskalen; die Projektordnung bleibt erhalten.",
            "Keine Cluster-, Signifikanz- oder Gewinnerinterpretation.",
        ),
        _record(
            "app_f2_exact_tie_bound_intervals",
            "Exakte Tie-Permutationsintervalle",
            "appendix",
            "data/figures/app_exact_tie_intervals.csv",
            "Welche exakten Min–Max-Werte sind durch Permutationen des Cutoff-Tieblocks möglich?",
            "Seedweise exakte Grenzen für p-only und Amount-Gain bei k=20,50,100",
            "Intervalle sind technische Permutationsgrenzen und keine Konfidenzintervalle.",
            "Keine statistische Unsicherheits- oder Signifikanzinterpretation.",
        ),
        _record(
            "app_f3_candidate_pool_ceiling_utilization",
            "Nutzung der Kandidatenpool-Ceilings",
            "appendix",
            "data/figures/app_candidate_pool_ceiling.csv",
            "Wie viel der im BCE-Top-1000-Pool verfügbaren Fraud- und PLR-Ceilings nutzt Amount-Gain?",
            "Seedweise post-hoc Verfügbarkeitsdiagnose mit Outer-Test-Labels",
            "Ceilings sind Verfügbarkeitsgrenzen, keine erwartete Modellleistung; Outer-Test-Labels werden nur post hoc verwendet.",
            "Keine erreichbare Erwartungsleistung, Trainingsinformation oder Oracle-Empfehlung.",
        ),
        _record(
            "ch5_t1_central_topk_results",
            "Zentrale Top-k-Ergebnisse",
            "main-text",
            "data/tables/ch5_t1_central_topk_results.csv",
            "Wie lauten PLR, Fraud, Precision und Recall an den zentralen Budgets?",
            "Arithmetisches Mittel ± Stichproben-SD über n=5 Seeds",
            "Drei Budgetpanels; PLR@k bezeichnet Fraud-Amount-Proxy-Abdeckung.",
            "Keine numerische Gewinnerhervorhebung oder Signifikanzrangfolge.",
        ),
        _record(
            "ch5_t2_seedwise_k50_diagnostic",
            "Seedweise k=50-Diagnose",
            "main-text",
            "data/tables/ch5_t2_seedwise_k50_diagnostic.csv",
            "Welche gepaarten Effekte, Ceiling-Nutzungen und Tie-Bounds liegen je Seed bei k=50 vor?",
            "Unaggregierte fünf Seedzeilen",
            "Tie-Ranges sind exakte technische Bereiche; Ceilings sind Verfügbarkeitsgrenzen.",
            "Keine Konfidenzintervalle oder erwartete Modellleistung.",
        ),
        _record(
            "ch5_t3_replacement_boundary_k50",
            "Replacement-Sets und Boundary-Events",
            "main-text",
            "data/tables/ch5_t3_replacement_summary.csv",
            "Wie unterscheiden sich vollständige Replacement-Sets und gepoolte Boundary-Events?",
            "Panel A: Seedmittel ± Stichproben-SD vollständiger Sets; Panel B: gepoolte Events mit BCE- oder Amount-Gain-Rang 30–70",
            "Die Panels verwenden bewusst verschiedene Aggregationsbasen und werden nicht zusammengeführt.",
            "Keine Vermischung von Seedmittel und gepoolter Fallebene.",
            additional=(
                "data/tables/ch5_t3_replacement_seedwise.csv",
                "data/tables/ch5_t3_boundary_pooled.csv",
            ),
        ),
        _record(
            "ch5_t4_high_amount_legit_k50",
            "High-Amount-Legit-Guardrail",
            "main-text",
            "data/tables/ch5_t4_high_amount_legit_summary.csv",
            "Welche legitimen und High-Amount-legitimen Fälle selektiert Amount-Gain@50?",
            "Kennzahlen je Seed; Mittel ± Stichproben-SD über n=5",
            "High-Amount-Schwelle ist das seedweise q90 legitimer Amounts.",
            "Keine Schadens-, Fairness- oder Kausalitätsaussage.",
            additional=(
                "data/tables/ch5_t4_high_amount_legit_seedwise.csv",
            ),
        ),
        _record(
            "app_t1_hard_impact_exact_values",
            "Exakte Hard-Impact-Werte",
            "appendix",
            "data/tables/app_t1_hard_impact_exact_values.csv",
            "Welche Seedwerte liegen dem Hard-Impact-Profil zugrunde?",
            "Unaggregierte Seed-Pfad-Werte bei q=0,90 und k=50",
            "BCE Miss Recovery ist nicht anwendbar.",
            "Keine Signifikanz- oder Realverlustaussage.",
        ),
        _record(
            "app_t2_global_metrics_by_budget",
            "Globale Ordnungsmetriken nach Budgetmodell",
            "appendix",
            "data/tables/app_t2_global_metrics_by_budget.csv",
            "Wie lauten ROC-AUC und AP je zentralem Budgetmodell?",
            "Arithmetisches Seedmittel ± Stichproben-SD",
            "Vollständige Testordnung; Nicht-BCE-Scores sind ordinal. Brier erscheint ausschließlich für BCE.",
            "Rankerscores sind keine kalibrierten Wahrscheinlichkeiten.",
        ),
        _record(
            "app_t3_exact_tie_bounds",
            "Exakte technische Tie-Bounds",
            "appendix",
            "data/tables/app_t3_exact_tie_bounds.csv",
            "Welche tatsächlichen und exakten Min–Max-Werte liegen je Tieblock vor?",
            "Unaggregierte technische Seedintervalle",
            "Permutationsgrenzen, keine Konfidenzintervalle.",
            "Keine statistische Unsicherheitsinterpretation.",
        ),
        _record(
            "app_t4_candidate_pool_coverage",
            "Kandidatenpool-Abdeckung und Ceiling-Nutzung",
            "appendix",
            "data/tables/app_t4_candidate_pool_coverage.csv",
            "Welche Fraud-/Amount-Abdeckung und Ceiling-Nutzung weist der Pool auf?",
            "Seedweise Amount-Gain-Diagnose bei k=20,50,100",
            "Ceilings sind post-hoc Verfügbarkeitsgrenzen.",
            "Keine erwartete Modellleistung oder Oracle-Vorgabe.",
        ),
        _record(
            "app_t5_seedwise_central_results",
            "Seedweise zentrale Resultate",
            "appendix",
            "data/tables/app_t5_seedwise_central_results.csv",
            "Welche exakten Seedwerte liegen den zentralen Resultaten zugrunde?",
            "Unaggregierte 5×3×4 Seed-Budget-Pfad-Zeilen",
            "Jeder trainierte Pfad gehört zu seinem budgetkonditionierten Modell.",
            "Keine Winner-Rangfolge oder Signifikanzspalte.",
        ),
    ]
    if tuple(row["artifact_id"] for row in artifacts) != CANONICAL_ARTIFACT_IDS:
        raise RuntimeError("Canonical presentation catalog inventory drift.")
    return {
        "schema": "fraud_detection.chapter5_presentation_selection.r7b.v1",
        "status": "PASS",
        "artifacts": artifacts,
    }


def engineering_evidence_statement(
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> str:
    """Return the explicit non-evidentiary boundary for an engineering run."""

    expected = {
        "mini-real": (
            "engineering mini profile — not thesis evidence",
            "real",
            "engineering mini profile — not thesis evidence",
        ),
        "smoke-synthetic": (
            "non-evidentiary",
            "synthetic",
            "synthetic engineering data — not thesis evidence",
        ),
    }
    if profile == "canonical":
        raise RuntimeError(
            "Engineering presentation catalog requested for canonical profile."
        )
    if profile not in expected:
        raise RuntimeError(f"Unsupported engineering profile: {profile!r}.")
    expected_evidence, expected_source, statement = expected[profile]
    if (
        evidence_classification != expected_evidence
        or data_source_kind != expected_source
    ):
        raise RuntimeError("Engineering presentation evidence metadata is missing.")
    return f"{statement}; {ENGINEERING_COMPARABILITY_BOUNDARY}"


def build_engineering_selection_registry(
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> dict[str, object]:
    """Return the shared two-artifact engineering acceptance catalog."""

    evidence_statement = engineering_evidence_statement(
        profile=profile,
        evidence_classification=evidence_classification,
        data_source_kind=data_source_kind,
    )
    shared = {
        "presentation_role": "engineering",
        "source_profile": profile,
        "evidence_classification": evidence_classification,
        "data_source_kind": data_source_kind,
        "evidence_statement": evidence_statement,
        "comparability_boundary": ENGINEERING_COMPARABILITY_BOUNDARY,
    }
    artifacts = [
        _record(
            ENGINEERING_ARTIFACT_IDS[0],
            "Engineering seed-budget delta heatmap",
            "engineering-acceptance",
            (
                "data/engineering/figures/"
                "engineering_seed_budget_delta_heatmap.csv"
            ),
            "Do all configured seed-budget-path deltas reach a figure renderer?",
            "Configured seed-budget-path values without evidentiary inference",
            evidence_statement,
            (
                "Technical acceptance only; not thesis evidence and "
                f"{ENGINEERING_COMPARABILITY_BOUNDARY}."
            ),
            **shared,
        ),
        _record(
            ENGINEERING_ARTIFACT_IDS[1],
            "Engineering central Top-k summary",
            "engineering-acceptance",
            (
                "data/engineering/tables/"
                "engineering_central_topk_summary.csv"
            ),
            "Do all configured central-budget path summaries reach a table renderer?",
            "Arithmetic seed mean and sample SD with ddof=1",
            evidence_statement,
            (
                "Technical acceptance only; no winner, significance, or thesis "
                "interpretation."
            ),
            **shared,
        ),
    ]
    if tuple(row["artifact_id"] for row in artifacts) != ENGINEERING_ARTIFACT_IDS:
        raise RuntimeError("Engineering presentation catalog inventory drift.")
    return {
        "schema": "fraud_detection.engineering_presentation_selection.v1",
        "status": "PASS",
        **shared,
        "artifacts": artifacts,
    }


def build_profile_selection_registry(
    *,
    presentation_role: str,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> dict[str, object]:
    """Select the canonical or engineering catalog from validated metadata."""

    if presentation_role == "canonical":
        if profile != "canonical":
            raise RuntimeError(
                "Canonical presentation catalog requested for noncanonical profile."
            )
        if (
            evidence_classification != "thesis-evidentiary"
            or data_source_kind != "real"
        ):
            raise RuntimeError("Canonical presentation evidence metadata is missing.")
        return build_selection_registry()
    if presentation_role == "engineering":
        return build_engineering_selection_registry(
            profile=profile,
            evidence_classification=evidence_classification,
            data_source_kind=data_source_kind,
        )
    raise RuntimeError(f"Unsupported presentation role: {presentation_role!r}.")
