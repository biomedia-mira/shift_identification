import pandas as pd
from pathlib import Path
from experiments import shift_generator

from experiments.inference_utils import get_or_save_outputs, get_ids_from_model_names
from shift_identification_detection.shift_identification import (
    run_multi_detection_identification,
    ALL_SHIFTS,
)

from torch.utils.data import DataLoader
import torch
import numpy as np
from data_handling.xray import RNSAPneumoniaDetectionDataset


def run_rsna(model_to_evaluate, encoder_to_evaluate, shift):
    n_cls = 2
    n_boostraps = 200
    val_sizes = [2000]
    test_sizes = [100, 250, 500, 1000]

    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    print(model_id, encoder_id)

    ### Create dataloaders
    val_df = pd.read_csv("val_rsna.csv")
    test_df = pd.read_csv("test_rsna.csv")
    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))

    val_dataset = RNSAPneumoniaDetectionDataset(
        df=val_df, transform=torch.nn.Identity()
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_dataset = RNSAPneumoniaDetectionDataset(
        df=test_df, transform=torch.nn.Identity()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ### Load model outputs (test + val)
    task_output, encoder_output = get_or_save_outputs(
        model_to_evaluate=model_to_evaluate,
        encoder_to_evaluate=encoder_to_evaluate,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        dataset_name="RSNA",
    )

    ### Define reference set sampling function
    def reference_set_sampling_fn(reference_set_size):
        return shift_generator.simple_val_sampling_base(val_df, reference_set_size)

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
            )
            res.to_csv(filename)
            print(f"Saved {filename}")

    ### Define shifted target set sampling function
    match shift:
        case "no_shift":
            if encoder_to_evaluate == "simclr_imagenet":
                print("Start no shift")
                prevalence_orig = 0.23
                filename = f"outputs/rsna_prev_{prevalence_orig}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.rsna_prev_shift(
                        test_df,
                        target_prevalence=prevalence_orig,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)
                print("Done no shift shift")

        case "prevalence":
            print("Start prevalence shift")
            prevalences = [0.10, 0.5, 0.80]
            for prevalence_shifted in prevalences:
                filename = f"outputs/rsna_prev_{prevalence_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.rsna_prev_shift(
                        test_df,
                        target_prevalence=prevalence_shifted,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "gender":
            print("Start gender shift")
            female_proportions = [0.25, 0.75, 1.0]
            for female_shifted in female_proportions:
                filename = f"outputs/rsna_gender_{female_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.rsna_gender_shift(
                        test_df,
                        target_female_proportion=female_shifted,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "gender_prev":
            print("Start gender + prev shifts")
            prev_gender_proportions = [(0.1, 0.75), (0.5, 0.75), (0.1, 1.0)]
            for disease_prev, female_shifted in prev_gender_proportions:
                filename = f"outputs/rsna_gender_prev_{female_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_prev{disease_prev}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.rsna_gender_and_prev_shift(
                        test_df,
                        target_female_proportion=female_shifted,
                        target_prevalence=disease_prev,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case _:
            print(f"Case {shift} is not defined for RSNA")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type")
    parser.add_argument("--shift", default="all")
    args = parser.parse_args()
    print(args)

    model_to_evaluate = (
        "/vol/biomedic3/mb121/shift_identification/outputs/run_j0c09xra/best.ckpt"
    )

    if args.encoder_type == "simclr_modality_specific":
        encoder_to_evaluate = (
            "/vol/biomedic3/mb121/causal-contrastive/outputs/run_q0kry6pk/best.ckpt"
        )
    elif args.encoder_type == "model":
        encoder_to_evaluate = model_to_evaluate
    else:
        encoder_to_evaluate = args.encoder_type
    if args.shift == "all":
        for shift in ALL_SHIFTS:
            run_rsna(model_to_evaluate, encoder_to_evaluate, shift)
    else:
        run_rsna(model_to_evaluate, encoder_to_evaluate, args.shift)
