

HPLT_LANGS = [
    "bos", "deu", "ron", "ell", "fin", "cat", "dan", "bul", "eng", "est", "eus",
    "fra", "hrv", "hun", "ita", "lit", "lvs", "mkd", "nld", "nno", "nob", "pol",
    "por", "slk", "slv", "spa", "swe", "ces", "tur", "ukr", "ara", "hin", "kor",
    "rus", "glg", "zhos", "zhot", "isl"
]

HPLT_TO_CODE_MAP = {
    "bos": "bos_Latn",
    "deu": "deu_Latn",
    "ron": "ron_Latn",
    "ell": "ell_Grek",
    "fin": "fin_Latn",
    "cat": "cat_Latn",
    "dan": "dan_Latn",
    "bul": "bul_Cyrl",
    "eng": "eng_Latn",
    "est": "est_Latn",
    "eus": "eus_Latn",
    "fra": "fra_Latn",
    "hrv": "hrv_Latn",
    "hun": "hun_Latn",
    "ita": "ita_Latn",
    "lit": "lit_Latn",
    "lvs": "lvs_Latn",
    "mkd": "mkd_Latn",
    "nld": "nld_Latn",
    "nno": "nno_Latn",
    "nob": "nob_Latn",
    "pol": "pol_Latn",
    "por": "por_Latn",
    "slk": "slk_Latn",
    "slv": "slv_Latn",
    "spa": "spa_Latn",
    "swe": "swe_Latn",
    "ces": "ces_Latn",
    "tur": "tur_Latn",
    "ukr": "ukr_Latn",
    "ara": "ara_Arab",
    "hin": "hin_Deva",
    "kor": "kor_Hang",
    "rus": "rus_Cyrl",
    "glg": "glg_Latn",
    "zhos": "zhos_Hans",
    "zhot": "zhot_Hant",
    "isl": "isl_Latn",
}

FEWSHOT=5

TEST_LANGS = [
    "ara", "deu", "eng", "fra", "hin", "ita", "nld", "por", "rus", "spa", "swe", "tur", "zhos"
]

TASK_LIST = [
    "belebele", #5, RC -- good
    # "xnli", #5, NLU
    # "arc", #5, GK -- bad
    "xcsqa", #5, RES -- good
    "hellaswag", #0, NLU -- good
    "flores200", #5, NLG -- good
    # "mmlu-ca", #5, GK
    # "mmlu-cs", #5, GK
    # "xquad", #5, RC (GEN),
    # "mkqa", #5, RC (GEN),
    # "xcodah" #5, RES,
]


TASK_PER_LANG = {
    "belebele": {
        # works
        "ara": "lighteval|belebele_arb_Arab_cf",
        "deu": "lighteval|belebele_deu_Latn_cf",
        "eng": "lighteval|belebele_eng_Latn_cf",
        "fra": "lighteval|belebele_fra_Latn_cf",
        "hin": "lighteval|belebele_hin_Deva_cf",
        "ita": "lighteval|belebele_ita_Latn_cf",
        "nld": "lighteval|belebele_nld_Latn_cf",
        "por": "lighteval|belebele_por_Latn_cf",
        "rus": "lighteval|belebele_rus_Cyrl_cf",
        "spa": "lighteval|belebele_spa_Latn_cf",
        "swe": "lighteval|belebele_swe_Latn_cf",
        "tur": "lighteval|belebele_tur_Latn_cf",
        "zhos": "lighteval|belebele_zho_Hans_cf",
    },
    "flores200": {
        # works
        "ara": "lighteval|flores200:eng_Latn-arb_Arab",
        "deu": "lighteval|flores200:eng_Latn-deu_Latn",
        "eng": "lighteval|flores200:fra_Latn-eng_Latn",
        "fra": "lighteval|flores200:eng_Latn-fra_Latn",
        "hin": "lighteval|flores200:eng_Latn-hin_Deva",
        "ita": "lighteval|flores200:eng_Latn-ita_Latn",
        "nld": "lighteval|flores200:eng_Latn-nld_Latn",
        "por": "lighteval|flores200:eng_Latn-por_Latn",
        "rus": "lighteval|flores200:eng_Latn-rus_Cyrl",
        "spa": "lighteval|flores200:eng_Latn-spa_Latn",
        "swe": "lighteval|flores200:eng_Latn-swe_Latn",
        "tur": "lighteval|flores200:eng_Latn-tur_Latn",
        "zhos": "lighteval|flores200:eng_Latn-zho_Hans",
    },
    "hellaswag": {
        # works
        "ara": "lighteval|mlmm_hellaswag_ara_cf",
        "deu": "lighteval|mlmm_hellaswag_deu_cf",
        "eng": "leaderboard|hellaswag",
        "fra": "lighteval|mlmm_hellaswag_fra_cf",
        "hin": "lighteval|mlmm_hellaswag_hin_cf",
        "ita": "lighteval|mlmm_hellaswag_ita_cf",
        "nld": "lighteval|mlmm_hellaswag_nld_cf",
        "por": "lighteval|mlmm_hellaswag_por_cf",
        "rus": "lighteval|mlmm_hellaswag_rus_cf",
        "spa": "lighteval|mlmm_hellaswag_spa_cf",
        "swe": "lighteval|mlmm_hellaswag_swe_cf",
        "tur": "lighteval|community_hellaswag_tur_cf",
        "zhos": "lighteval|mlmm_hellaswag_zho_cf",
    },
    "xnli": {
        # works
        "ara": "lighteval|xnli2.0_ara_cf",
        "deu": "lighteval|xnli2.0_deu_cf",
        "eng": "lighteval|xnli_eng_cf",
        "fra": "lighteval|xnli2.0_fra_cf",
        "hin": "lighteval|xnli2.0_hin_cf",
        "ita": None,
        "nld": None,
        "por": None,
        "rus": "lighteval|xnli2.0_rus_cf",
        "spa": "lighteval|xnli2.0_spa_cf",
        "tur": "lighteval|xnli2.0_tur_cf",
        "zhos": "lighteval|xnli2.0_zho_cf",
    },
    "arc": {
        # works
        "ara": "lighteval|mlmm_arc_ara_cf:challenge",
        "deu": "lighteval|mlmm_arc_deu_cf:challenge",
        "eng": "leaderboard|arc:challenge",
        "fra": "lighteval|mlmm_arc_fra_cf:challenge",
        "hin": "lighteval|community_arc_hin_cf:challenge",
        "ita": "lighteval|mlmm_arc_ita_cf:challenge",
        "nld": "lighteval|mlmm_arc_nld_cf:challenge",
        "por": None,
        "rus": "lighteval|mlmm_arc_rus_cf:challenge",
        "spa": "lighteval|mlmm_arc_spa_cf:challenge",
        "tur": "lighteval|community_arc_tur_cf:challenge",
        "zhos": "lighteval|mlmm_arc_zho_cf:challenge",
    },
    "xcsqa": {
        # works
        # xcodah_zh fails for some reason
        "ara": "lighteval|xcsqa_ara_cf",
        "deu": "lighteval|xcsqa_deu_cf",
        "eng": "lighteval|xcsqa_eng_cf",
        "fra": "lighteval|xcsqa_fra_cf",
        "hin": "lighteval|xcsqa_hin_cf",
        "ita": "lighteval|xcsqa_ita_cf",
        "nld": "lighteval|xcsqa_nld_cf",
        "por": "lighteval|xcsqa_por_cf",
        "rus": "lighteval|xcsqa_rus_cf",
        "spa": "lighteval|xcsqa_spa_cf",
        "swe": None,
        "tur": None,
        "zhos": "lighteval|xcsqa_zho_cf",
    },
    "xcodah": {
        # works
        "ara": "lighteval|xcodah_ara_cf",
        "deu": "lighteval|xcodah_deu_cf",
        "eng": "lighteval|xcodah_eng_cf",
        "fra": "lighteval|xcodah_fra_cf",
        "hin": "lighteval|xcodah_hin_cf",
        "ita": "lighteval|xcodah_ita_cf",
        "nld": "lighteval|xcodah_nld_cf",
        "por": "lighteval|xcodah_por_cf",
        "rus": "lighteval|xcodah_rus_cf",
        "spa": "lighteval|xcodah_spa_cf",
        "swe": None,
        "tur": None,
        "zhos": "lighteval|xcodah_zho_cf",
    },
    "mkqa": {
        # works
        "ara": "lighteval|mkqa_ara",
        "deu": "lighteval|mkqa_deu",
        "eng": "lighteval|mkqa_eng",
        "fra": "lighteval|mkqa_fra",
        "hin": None,
        "ita": "lighteval|mkqa_ita",
        "nld": "lighteval|mkqa_nld",
        "por": "lighteval|mkqa_por",
        "rus": "lighteval|mkqa_rus",
        "spa": "lighteval|mkqa_spa",
        "tur": "lighteval|mkqa_tur",
        "zhos": "lighteval|mkqa_zho",
    },
    "mmlu-ca": {
        # works, but mmlu_all fails
        "ara": "lighteval|global_mmlu_ca_ara_cf",
        "deu": "lighteval|global_mmlu_ca_deu_cf",
        "eng": "lighteval|global_mmlu_ca_eng_cf",
        "fra": "lighteval|global_mmlu_ca_fra_cf",
        "hin": "lighteval|global_mmlu_ca_hin_cf",
        "ita": "lighteval|global_mmlu_ca_ita_cf",
        "nld": "lighteval|global_mmlu_ca_nld_cf",
        "por": "lighteval|global_mmlu_ca_por_cf",
        "rus": "lighteval|global_mmlu_ca_rus_cf",
        "spa": "lighteval|global_mmlu_ca_spa_cf",
        "swe": "lighteval|global_mmlu_ca_swe_cf",
        "tur": "lighteval|global_mmlu_ca_tur_cf",
        "zhos": "lighteval|global_mmlu_ca_zho_cf",
    },
    "mmlu-cs": {
        # works
        "ara": "lighteval|global_mmlu_cs_ara_cf",
        "deu": "lighteval|global_mmlu_cs_deu_cf",
        "eng": "lighteval|global_mmlu_cs_eng_cf",
        "fra": "lighteval|global_mmlu_cs_fra_cf",
        "hin": "lighteval|global_mmlu_cs_hin_cf",
        "ita": "lighteval|global_mmlu_cs_ita_cf",
        "nld": "lighteval|global_mmlu_cs_nld_cf",
        "por": "lighteval|global_mmlu_cs_por_cf",
        "rus": "lighteval|global_mmlu_cs_rus_cf",
        "spa": "lighteval|global_mmlu_cs_spa_cf",
        "swe": "lighteval|global_mmlu_cs_swe_cf",
        "tur": "lighteval|global_mmlu_cs_tur_cf",
        "zhos": "lighteval|global_mmlu_cs_zho_cf",
    },
    "xquad": {
        # works
        "ara": "lighteval|xquad_ara",
        "deu": "lighteval|xquad_deu",
        "eng": "lighteval|xquad_eng",
        "fra": "lighteval|fquadv2_fra",
        "hin": "lighteval|xquad_hin",
        "ita": "lighteval|squad_ita",
        "nld": None,
        "por": None,
        "rus": "lighteval|xquad_rus",
        "spa": "lighteval|xquad_spa",
        "tur": "lighteval|tquadv2_tur",
        "zhos": "lighteval|xquad_zho",
    },
}


# Derived: all unique language codes across all tasks
ALL_LANGS: list[str] = sorted({
    lang
    for lang_map in TASK_PER_LANG.values()
    for lang in lang_map
})

# Tasks where the metric is generative (chrf++/bleu) rather than acc_norm
GENERATIVE_TASKS = {"flores200", "mkqa", "xquad"}

# The fewshot suffix appended to each key in the JSON (task -> suffix)
FEWSHOT_SUFFIX: dict[str, str] = {
    "belebele":  "|5",
    "flores200": "|5",
    "hellaswag": "|5",
    "xnli":      "|5",
    "arc":       "|5",
    "xcsqa":     "|5",
    "xcodah":    "|5",
    "mkqa":      "|5",
    "mmlu-ca":   "|5",
}

# For mmlu-ca we look up the pre-aggregated _average subtask key
MMLU_AVG_SUFFIX = ":_average"


def task_json_key(task: str, lang: str) -> str | None:
    """
    Return the full JSON key for (task, lang), or None if unavailable.
    Handles the mmlu-ca _average suffix and fewshot suffix automatically.
    """
    prefix = TASK_PER_LANG.get(task, {}).get(lang)
    if prefix is None:
        return None
    suffix = FEWSHOT_SUFFIX.get(task, "|5")
    if task == "mmlu-ca":
        return f"{prefix}{MMLU_AVG_SUFFIX}{suffix}"
    return f"{prefix}{suffix}"


def metric_for_task(task: str) -> tuple[str, str]:
    """Return (metric_key, stderr_key) appropriate for this task."""
    if task in GENERATIVE_TASKS:
        return "chrf++", "chrf++_stderr"
    return "acc_norm", "acc_norm_stderr"