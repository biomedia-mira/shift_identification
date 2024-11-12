import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


def _histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


def cross_entropy_loss(logits, labels):
    loss = -np.mean(logits[np.arange(len(labels)), labels])
    return loss


def add_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def ece_loss(probs, labels, num_bins=15, equal_mass=True):
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = np.equal(predictions, labels)

    if not equal_mass:
        bins = np.linspace(0, 1, num_bins + 1)
    else:
        bins = _histedges_equalN(confidences, num_bins)

    ece = 0.0

    for i in range(num_bins):
        in_bin = np.greater_equal(confidences, bins[i]) & np.less(
            confidences, bins[i + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(in_bin * accuracies)
            avg_confidence_in_bin = np.mean(in_bin * confidences)
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# From https://github.com/saurabhgarg1996/calibration
# Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py


class VectorScaling:
    def __init__(
        self, num_label, bias=False, weights=None, device=None, print_verbose=False
    ):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.num_label = num_label

        self.temperature = nn.Parameter(torch.ones(num_label).to(self.device) * 1.5)
        self.bias = nn.Parameter((torch.rand(num_label) * 2.0 - 1.0).to(device))

        self.biasFlag = bias
        self.print_verbose = print_verbose

        if weights is not None:
            self.weights = weights.to(device)
        else:
            self.weights = None

    def forward(self, input):
        return self.temperature_scale(input)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits

        if self.biasFlag:
            bias = self.bias.unsqueeze(0).expand(logits.size(0), -1)
            return logits * torch.exp(self.temperature) + bias
        else:
            return logits * torch.exp(self.temperature)

    def fit(self, logits, labels, eps=1e-12):
        # First: collect all the logits and labels for the validation set
        before_temperature_nll = cross_entropy_loss(logits, labels)
        probs = add_softmax(logits, axis=-1)
        before_temperature_ece = ece_loss(probs, labels)

        if self.print_verbose:
            print(
                "Before temperature - NLL: %.3f, ECE: %.3f"
                % (before_temperature_nll, before_temperature_ece)
            )

        torch_labels = torch.from_numpy(labels).long().to(self.device)
        torch_logits = torch.from_numpy(logits).float().to(self.device)

        if self.weights is not None:
            nll_criterion = nn.CrossEntropyLoss(weight=self.weights)
        else:
            nll_criterion = nn.CrossEntropyLoss()

        # Next: optimize the temperature w.r.t. NLL
        if not self.biasFlag:
            optimizer = optim.LBFGS([self.temperature])

        else:
            optimizer = optim.LBFGS([self.temperature, self.bias])

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(torch_logits), torch_labels)
            loss.backward()
            return loss

        loss = -10.0
        new_loss = -1.0

        run = True
        count = 0
        while run:
            while np.abs(loss - new_loss) > 1e-4:
                loss = new_loss
                if self.print_verbose:
                    print(f"Loss : {loss}")

                optimizer.step(eval)

                with torch.no_grad():
                    new_loss = (
                        nll_criterion(
                            self.temperature_scale(torch_logits), torch_labels
                        )
                        .cpu()
                        .numpy()
                    )

            if (torch.isnan(self.temperature)).any() or (torch.isnan(self.bias)).any():
                self.temperature = nn.Parameter(
                    torch.ones(self.num_label).to(self.device) * 1.5
                )
                self.bias = nn.Parameter(
                    (torch.rand(self.num_label) * 2.0 - 1.0).to(self.device)
                )
                run = True
                count += 1
            else:
                run = False

            if count > 10:
                run = False

        torch_logits = self.temperature_scale(torch_logits)
        rescaled_probs = F.softmax(torch_logits, dim=-1).detach().cpu().numpy()

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = cross_entropy_loss(
            torch_logits.detach().cpu().numpy(), labels
        )
        after_temperature_ece = ece_loss(rescaled_probs, labels)

        if self.print_verbose:
            print("Optimal temperature: ", self.temperature.detach().cpu().numpy())
            if self.biasFlag:
                print("Optimal bias: ", self.bias.detach().cpu().numpy())

            print(
                "After temperature - NLL: %.3f, ECE: %.3f"
                % (after_temperature_nll, after_temperature_ece)
            )

    def calibrate(self, logits, eps=1e-12):
        torch_logits = torch.from_numpy(logits).float().to(self.device)
        torch_logits = self.temperature_scale(torch_logits).detach().cpu()
        rescaled_probs = F.softmax(torch_logits, dim=-1).detach().cpu().numpy()

        return rescaled_probs
