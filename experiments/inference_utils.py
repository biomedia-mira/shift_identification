import torch

from pathlib import Path
import pickle

from classification.classification_module import ClassificationModule
from tqdm.autonotebook import tqdm

from classification.vit_models import mae_vit_base_patch16
from torchvision.transforms.functional import center_crop


def get_ids_from_model_names(encoder_name, model_name):
    encoder_id = (
        Path(encoder_name).parent.stem[4:]
        if (
            "imagenet" not in encoder_name
            and encoder_name not in ["raddino", "cxr_mae", "imagenet_mae", "random"]
        )
        else encoder_name
    )

    model_id = Path(model_name).parent.stem[4:]
    return encoder_id, model_id


def get_or_save_outputs(
    model_to_evaluate, encoder_to_evaluate, val_loader, test_loader, dataset_name
):
    """
    Inference loop. If already saved simply returns dictionary of outputs.
    Else computes results for task model and encoder model, saves and returns the results.
    """
    encoder_id, model_id = get_ids_from_model_names(
        encoder_to_evaluate, model_to_evaluate
    )
    outputs_dir = Path(f"outputs/{dataset_name}")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_filename = outputs_dir / f"model_{model_id}.pkl"
    print(encoder_id)
    encoder_filename = outputs_dir / f"encoder_{encoder_id}.pkl"
    print(model_filename, encoder_filename)
    compute_task, compute_encoder = True, True
    if model_filename.exists():
        with open(str(model_filename), "rb") as fp:
            task_output = pickle.load(fp)
            compute_task = False
    else:
        task_output = {}
        model = ClassificationModule.load_from_checkpoint(
            model_to_evaluate, map_location="cuda:0", strict=False
        ).model.eval()
        model.cuda()

    if encoder_filename.exists():
        with open(str(encoder_filename), "rb") as fp:
            encoder_output = pickle.load(fp)
            compute_encoder = False
    else:
        encoder_output = {}
        match encoder_to_evaluate:
            case "imagenet":
                encoder = ClassificationModule(
                    num_classes=2,
                    encoder_name="resnet50",
                    input_channels=1 if dataset_name != "Retina" else 3,
                    pretrained=True,
                ).model.eval()
            case "simclr_imagenet":
                # From https://github.com/AndrewAtanov/simclr-pytorch/blob/master/README.md
                model_weights = "pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar"  # noqa
                # Converting state dict to my model wrapper
                state_dict = torch.load(model_weights)["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if "fc" not in k and "projection" not in k:
                        new_state_dict[k.replace("convnet.", "model.net.")] = v
                encoder_module = ClassificationModule(
                    num_classes=2, encoder_name="resnet50", input_channels=3
                )
                encoder_module.load_state_dict(new_state_dict, strict=False)
                encoder = encoder_module.model.eval()
            case _:
                try:
                    encoder = ClassificationModule.load_from_checkpoint(
                        encoder_to_evaluate,
                        map_location="cuda:0",
                        strict=False,
                        encoder_name="resnet50",
                    ).model.eval()
                except RuntimeError:
                    encoder = ClassificationModule.load_from_checkpoint(
                        encoder_to_evaluate,
                        map_location="cuda:0",
                        strict=False,
                        encoder_name="resnet18",
                    ).model.eval()
        encoder = encoder.to("cuda")

    if compute_task or compute_encoder:
        for name, loader in [("val", val_loader), ("test", test_loader)]:
            y_val = []
            probas = []
            encoder_feats = []
            encoder_early_feats = []
            with torch.no_grad():
                for batch in tqdm(loader):
                    x = batch["x"].cuda()
                    y = batch["y"]
                    y_val.append(y)
                    if compute_task:
                        probas.append(model(x).cpu())
                    if compute_encoder:
                        # EMBED data by default 224*192. Not compatible with imagenet mae resize to 224*224
                        if encoder_to_evaluate == "imagenet_mae" and x.shape[-1] != 224:
                            x = center_crop(x, 224)
                        try:
                            feats1, last_feats = encoder.get_features(
                                x, include_early_feats=True
                            )
                            encoder_early_feats.append(feats1.cpu())
                        except TypeError:
                            last_feats = encoder.get_features(x)

                        encoder_feats.append(last_feats.cpu())

            y_val = torch.concatenate(y_val)

            if compute_task:
                probas = torch.softmax(torch.concatenate(probas), 1)
                task_output.update(
                    {
                        name: {
                            "y": y_val,
                            "probas": probas,
                        }
                    }
                )
            if compute_encoder:
                encoder_feats = torch.concatenate(encoder_feats)
                if len(encoder_early_feats) > 0:
                    encoder_early_feats = torch.concatenate(encoder_early_feats)
                encoder_output.update(
                    {
                        name: {
                            "y": y_val,
                            "feats": encoder_feats,
                            "early_feats": encoder_early_feats,
                        }
                    }
                )
        if compute_encoder:
            with open(str(encoder_filename), "wb") as fp:
                pickle.dump(encoder_output, fp)
                print("dictionary saved successfully to file")
        if compute_task:
            with open(str(model_filename), "wb") as fp:
                pickle.dump(task_output, fp)
                print("dictionary saved successfully to file")
    return task_output, encoder_output
