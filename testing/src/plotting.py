import os
from src.utils import LABELS

def plot_probabilities(probs_accumulated, real_input, output_path="classifier_plot.tex"):
    """
    Write a TikZ/PGFPlots snippet plotting each class probability over samples.
    The output is a TeX file you can \input into your document.
    """
    num_samples, num_classes = probs_accumulated.shape
    with open(output_path, 'w') as f:
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[xlabel=Sample index, ylabel=Probability, grid=major, legend pos=outer north east]\n")
        for i, label in enumerate(LABELS):
            coords = " ".join(f"({j},{probs_accumulated[j,i]:.3f})" for j in range(num_samples))
            f.write(f"\\addplot coordinates {{ {coords} }};\\addlegendentry{{{label}}}\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")