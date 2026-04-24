import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from utils.langutils import TEST_LANGS, TASK_PER_LANG, TASK_LIST

import os
import torch
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs

os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["VLLM_USE_V1"] = "0"               # use stable vLLM v0 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation

# hf token in ./token
os.environ["HF_TOKEN"] = open("./token", "r").read().strip()

def model_output_name(model_path: str, save_name: str | None = None, revision: str | None = None) -> str:
    """
    Derive a clean result folder name from a model path.
    If save_name is provided (e.g. from --output_dir_suffix), use that instead.

    Examples:
      "HPLT/hplt2c_eng_checkpoints"                               -> "HPLT/hplt2c_eng_checkpoints"
      "/abs/path/models/mixed-10-checkpoints/checkpoint_0001000"  -> "mixed-10-checkpoints/checkpoint_0001000"
      save_name="checkpoint_0047684"                               -> "checkpoint_0047684"
    """
    if revision and revision == "main":
        revision = "checkpoint_0047684"

    # if HF model, ignore first part of model_path by counting if only one / in model_path
    if model_path.count("/") == 1:
        model_path = model_path.split("/")[1]

    if revision:
        if revision not in model_path:
            return f"{model_path}/{revision}"
    elif save_name:
        return save_name
    if not model_path.startswith("/"):
        return model_path  # HF hub ID, use as-is
    parts = model_path.rstrip("/").split("/")
    return "/".join(parts[-2:])


def main(args):

    print("Using GPUs:", torch.cuda.device_count())
    assert args.gpus <= torch.cuda.device_count()

    model_output_dir = model_output_name(args.model, args.output_dir_suffix, getattr(args, "revision", None))
    print("Output directory:", os.path.join(args.output_dir, model_output_dir))

    evaluation_tracker = EvaluationTracker(
        output_dir=os.path.join(args.output_dir, model_output_dir),
        save_details=False,
        push_to_hub=False,
        hub_results_org="sethjsa",
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="lighteval.tasks.multilingual.tasks",
        max_samples=950
    )

    if "HPLT" in args.model or "checkpoints" in args.model:
        max_model_length = 2047
    else:
        max_model_length = 2049

    model_config = VLLMModelConfig(
        model_name=args.model,
        **({"revision": args.revision} if args.revision is not None else {}),
        dtype="bfloat16",
        tensor_parallel_size=1,
        data_parallel_size=args.gpus,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_length=max_model_length,
        swap_space=2,
        seed=1,
        trust_remote_code=True,
        add_special_tokens=True,
        multichoice_continuations_start_space=True,
        pairwise_tokenization=True,
        generation_parameters={
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "min_p": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "stop_tokens": None,
            "max_new_tokens": 256,
            "min_new_tokens": 0,
        }
    )

    all_tasks = ""

    tasks = TASK_LIST if args.task == "all" else [args.task]
    langs = TEST_LANGS if args.lang == "all" else [args.lang]

    print("Model:", args.model)
    print("Revision:", args.revision)
    print("Output directory:", model_output_dir)
    print("Evaluating tasks:", tasks)
    print("Evaluating languages:", langs)

    for task in tasks:
        for lang in langs:
            lang_task = TASK_PER_LANG[task][lang]
            print(lang_task)
            if lang_task:
                all_tasks += f"{lang_task}|{args.few_shot}|{args.truncate_few_shot},"
    all_tasks = all_tasks.rstrip(",")

    pipeline = Pipeline(
        tasks=all_tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--few_shot", type=int, default=5)
    parser.add_argument("--truncate_few_shot", type=int, default=0)
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--model", type=str, default="HPLT/hplt2c_eng_checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--output_dir_suffix", type=str, default=None,
                        help="Override the derived output subfolder name (used for HF checkpoints saved under a custom name)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()
    main(args)