import pandas as pd
from pathlib import Path

from experiments.inference_utils import get_or_save_outputs, get_ids_from_model_names
from shift_identification_detection.shift_identification import (
    run_multi_detection_identification,
    ALL_SHIFTS,
)

from torch.utils.data import DataLoader
import torch
import numpy as np
from data_handling.retina import RetinaDataset
from experiments import shift_generator


def run_retina(model_to_evaluate, encoder_to_evaluate, shift):
    n_cls = 2
    n_boostraps = 200
    val_sizes = [1000]
    test_sizes = [100, 250, 500, 1000]

    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    print(model_id, encoder_id)

    ### Create dataloaders
    val_df = pd.read_csv("retina_val.csv")
    test_df = pd.read_csv("retina_test.csv")
    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))
    val_dataset = RetinaDataset(df=val_df, transform=torch.nn.Identity())
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=12
    )
    test_dataset = RetinaDataset(df=test_df, transform=torch.nn.Identity())
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=12
    )

    task_output, encoder_output = get_or_save_outputs(
        model_to_evaluate=model_to_evaluate,
        encoder_to_evaluate=encoder_to_evaluate,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        dataset_name="Retina",
    )

    ### Define reference set sampling function
    def reference_set_sampling_fn(reference_set_size):
        return shift_generator.retina_acq_prev_shift(
            val_df, target_dataset_size=reference_set_size
        )

    def _run_identification_if_necessary(filename, shifted_set_generating_fn):
        if Path(filename).exists():
            res = pd.read_csv(filename)
        else:
            res = run_multi_detection_identification(
                task_output,
                encoder_output,
                shifted_set_generating_fn,
                reference_set_sampling_fn,
                n_boostraps=n_boostraps,
                test_sizes=test_sizes,
                val_sizes=val_sizes,
                num_classes=n_cls,
                run_mmd_on_early_feats=(encoder_to_evaluate == model_to_evaluate),
            )
            res.to_csv(filename)
            print(f"Saved {filename}")

    match shift:
        case "prevalence":
            print("Start prevalence shift")
            for prevalence_shifted in [0.50, 0.65, 0.78, 1.0]:
                filename = f"outputs2/retina_prev_{prevalence_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.retina_acq_prev_shift(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_prevalence=prevalence_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "acquisition":
            print("Start acquisition shift")
            site_distributions = [
                [0.30, 0.30, 0.40],
                [0.05, 0.30, 0.65],
                [0.20, 0.20, 0.6],
            ]
            for site_shifted in site_distributions:
                filename = f"outputs2/retina_acq_{site_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.retina_acq_prev_shift(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_site_distribution=site_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "acquisition_prev":
            if encoder_to_evaluate == "simclr_imagenet":
                print("Start acquisition + prev shift")
                site_distributions = [
                    [0.30, 0.30, 0.40],
                    [0.05, 0.30, 0.65],
                    [0.20, 0.20, 0.6],
                ]
                prevalence_shifted = 0.5
                for site_shifted in site_distributions:
                    filename = f"outputs2/retina_acq_prev_{site_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_t{test_sizes}.csv"

                    def shifted_set_generating_fn(target_set_size):
                        return shift_generator.retina_acq_prev_shift(
                            test_df,
                            target_dataset_size=target_set_size,
                            target_site_distribution=site_shifted,
                            target_prevalence=prevalence_shifted,
                        )

                    _run_identification_if_necessary(
                        filename, shifted_set_generating_fn
                    )

        case _:
            print(f"Case {shift} is not defined for RETINA")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type")
    parser.add_argument("--shift", default="all")
    args = parser.parse_args()
    print(args)

    model_to_evaluate = (
        "/vol/biomedic3/mb121/shift_identification/outputs/run_ve3it5qy/best.ckpt"
    )
    if args.encoder_type == "simclr_modality_specific":
        encoder_to_evaluate = "/vol/biomedic3/mb121/causal-contrastive/outputs2/run_cwyi1g3d/epoch=449.ckpt"
    elif args.encoder_type == "model":
        encoder_to_evaluate = model_to_evaluate
    else:
        encoder_to_evaluate = args.encoder_type

    if args.shift == "all":
        for shift in ALL_SHIFTS:
            run_retina(model_to_evaluate, encoder_to_evaluate, shift)
    else:
        run_retina(model_to_evaluate, encoder_to_evaluate, args.shift)
