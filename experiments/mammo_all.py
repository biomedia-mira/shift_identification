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
from experiments import shift_generator
from data_handling.mammo import EmbedDataset


def run_mammo(model_to_evaluate, encoder_to_evaluate, shift):
    n_boostraps = 200
    val_size = 1000
    test_sizes = [250, 50, 100]
    n_cls = 4

    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    print(model_id, encoder_id)

    ### Create dataloaders
    val_df = pd.read_csv("val_embed.csv")
    test_df = pd.read_csv("test_embed.csv")

    test_df["idx_in_original"] = np.arange(len(test_df))
    val_df["idx_in_original"] = np.arange(len(val_df))

    val_dataset = EmbedDataset(df=val_df, transform=torch.nn.Identity(), cache=False)
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=12
    )
    test_dataset = EmbedDataset(df=test_df, transform=torch.nn.Identity(), cache=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=12
    )

    ### Load model outputs (test + val)
    task_output, encoder_output = get_or_save_outputs(
        model_to_evaluate=model_to_evaluate,
        encoder_to_evaluate=encoder_to_evaluate,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        dataset_name="Mammo",
    )

    ### Define reference set sampling function
    def reference_set_sampling_fn(reference_set_size):
        return shift_generator.simple_val_sampling_embed_stratified(
            val_df, reference_set_size
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
                val_sizes=[val_size],
                num_classes=n_cls,
                is_embed=True,
                run_mmd_on_early_feats=(encoder_to_evaluate == model_to_evaluate),
            )
            res.to_csv(filename)
            print(f"Saved {filename}")

    ### Define shifted target set sampling function
    match shift:
        case "no_shift":
            if encoder_to_evaluate == "simclr_imagenet":
                print("Start no shift")
                filename = f"outputs2/mammo_noshift_n{n_boostraps}_{model_id}_{encoder_id}_v{val_size}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.simple_val_sampling_embed_stratified(
                        test_df, target_set_size
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "prevalence":
            print("Start prevalence shift")
            prevalences = [
                [0.0, 0.5, 0.5, 0.0],
                [0.15, 0.35, 0.35, 0.15],
                [0.10, 0.20, 0.60, 0.10],
            ]
            for prevalence_shifted in prevalences:
                filename = f"outputs2/mammo_prev_{prevalence_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_size}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.mammo_acq_prev_shift(
                        test_df=test_df,
                        target_dataset_size=target_set_size,
                        target_density_distribution=prevalence_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "acquisition":
            print("Start acquisition shift")
            scanner_distributions = [
                [0.55, 0.00, 0.10, 0.10, 0.15, 0.10],
                [0.50, 0.00, 0.00, 0.20, 0.20, 0.10],
                [0.33, 0.02, 0.20, 0.15, 0.20, 0.10],
            ]
            for scanner_shifted in scanner_distributions:
                filename = f"outputs2/mammo_acq_{scanner_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_size}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.mammo_acq_prev_shift(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_manufacturer_distribution=scanner_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "acquisition_prev":
            print("Start acquisition + prev shift")
            scanner_distributions = [
                [0.33, 0.02, 0.20, 0.15, 0.20, 0.10],
                [0.50, 0.00, 0.00, 0.20, 0.20, 0.10],
                [0.55, 0.00, 0.10, 0.10, 0.15, 0.10],
            ]
            disease_prev = [0.0, 0.5, 0.5, 0.0]
            for scanner_shifted in scanner_distributions:
                print(scanner_shifted, disease_prev)
                filename = f"outputs2/mammo_acq_prev_{scanner_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_size}_prev{disease_prev}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.mammo_acq_prev_shift(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_manufacturer_distribution=scanner_shifted,
                        target_density_distribution=disease_prev,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type")
    parser.add_argument("--shift", default="all")
    args = parser.parse_args()
    print(args)

    model_to_evaluate = (
        "path-to-task-model-checkpoint"
    )

    if args.encoder_type == "simclr_modality_specific":
        encoder_to_evaluate = (
            "path-to-modality-specific-simclr-checkpoint"
        )
    elif args.encoder_type == "model":
        encoder_to_evaluate = model_to_evaluate
    else:
        encoder_to_evaluate = args.encoder_type
        print(args.shift)
    if args.shift == "all":
        for shift in ALL_SHIFTS:
            run_mammo(model_to_evaluate, encoder_to_evaluate, shift)
    else:
        run_mammo(model_to_evaluate, encoder_to_evaluate, args.shift)
