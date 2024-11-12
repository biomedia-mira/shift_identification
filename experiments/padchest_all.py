import pandas as pd
from pathlib import Path

from experiments.inference_utils import get_or_save_outputs, get_ids_from_model_names
from shift_identification.shift_identification import (
    run_multi_detection_identification,
    ALL_SHIFTS,
)
from torch.utils.data import DataLoader
import torch
import numpy as np
from data_handling.xray import PadChestDataset
from experiments import shift_generator


def run_padchest(model_to_evaluate, encoder_to_evaluate, shift):
    n_cls = 2
    n_boostraps = 200
    val_sizes = [2000]
    test_sizes = [100, 250, 500, 1000]

    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    print(model_id, encoder_id)

    ### Create dataloaders
    val_df = pd.read_csv("val_padchest.csv")
    test_df = pd.read_csv("test_padchest.csv")

    val_df["idx_in_original"] = np.arange(len(val_df))
    test_df["idx_in_original"] = np.arange(len(test_df))

    val_dataset = PadChestDataset(
        df=val_df, transform=torch.nn.Identity(), label_column="pneumonia"
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=12
    )
    test_dataset = PadChestDataset(
        df=test_df, transform=torch.nn.Identity(), label_column="pneumonia"
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=12
    )

    ### Load model outputs (test + val)
    task_output, encoder_output = get_or_save_outputs(
        model_to_evaluate=model_to_evaluate,
        encoder_to_evaluate=encoder_to_evaluate,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        dataset_name="PadChest",
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
                run_mmd_on_early_feats=(encoder_to_evaluate == model_to_evaluate),
            )
            res.to_csv(filename)
        print(f"Saved {filename}")

    ### Define shifted target set sampling function
    match shift:
        case "no_shift":
            if encoder_to_evaluate == "simclr_imagenet":
                print("Start no shift")
                prevalence_orig = 0.036

                filename = f"outputs2/padchest_prev_{prevalence_orig}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.sample_shift_padchest(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_pneumonia=prevalence_orig,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)
                print("Done no shift shift")

        case "acquisition":
            print("Start acquisition shift")
            for phillips_shifted in [0.25, 0.75, 1.0]:
                filename = f"outputs2/padchest_acq_{phillips_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.sample_shift_padchest(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_prev_phillips=phillips_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "gender":
            print("Start gender shift")
            for female_shifted in [0.25, 0.75, 1.0]:
                filename = f"outputs2/padchest_gender_{female_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.padchest_gender_shift(
                        test_df,
                        target_female_proportion=female_shifted,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "prevalence":
            print("Start prevalence shift")
            for prevalence_shifted in [0.15, 0.20, 0.30]:
                filename = f"outputs2/padchest_prev_{prevalence_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"
                print(prevalence_shifted)

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.sample_shift_padchest(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_pneumonia=prevalence_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)
            print("Done prevalence shift")

        case "gender_prev":
            print("Start gender + prev shift")
            female_proportions = [0.75, 0.25, 1.0]
            prevalence_shifted = 0.15
            for female_shifted in female_proportions:
                filename = f"outputs2/padchest_gender_prev_{female_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    shift_generator.padchest_gender_prev_shift(
                        test_df,
                        target_female_proportion=female_shifted,
                        target_disease=prevalence_shifted,
                        target_dataset_size=target_set_size,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case "acquisition_prev":
            print("Start acquisition + prev shift")
            for phillips_shifted, prevalence_shifted in [
                (0.75, 0.15),
                (0.75, 0.25),
                (1.0, 0.25),
            ]:
                filename = f"outputs2/padchest_acq_prev_{phillips_shifted}_n{n_boostraps}_{model_id}_{encoder_id}_v{val_sizes[0]}_d{prevalence_shifted}_t{test_sizes}.csv"

                def shifted_set_generating_fn(target_set_size):
                    return shift_generator.sample_shift_padchest(
                        test_df,
                        target_dataset_size=target_set_size,
                        target_prev_phillips=phillips_shifted,
                        target_pneumonia=prevalence_shifted,
                    )

                _run_identification_if_necessary(filename, shifted_set_generating_fn)

        case _:
            print(f"Case {shift} is not defined for PadChest")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type")
    parser.add_argument("--shift", default="all")
    args = parser.parse_args()
    print(args)

    model_to_evaluate = (
        "/vol/biomedic3/mb121/shift_identification/outputs/run_h4tbta6v/best.ckpt"
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
            run_padchest(model_to_evaluate, encoder_to_evaluate, shift)
    else:
        run_padchest(model_to_evaluate, encoder_to_evaluate, args.shift)
