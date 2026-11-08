"""Audit the executed Workshop 4 notebook and its saved artifacts."""

from __future__ import annotations

import hashlib
import base64
import re
from pathlib import Path

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "workshop4_autoencoders" / "notebooks" / "workshop4_autoencoders.ipynb"
README = ROOT / "workshop4_autoencoders" / "README.md"

EXPECTED_HASHES = {
    "data/splits.json": "cb2572a0e61676c02c5957107a59975353cbaffd2750f5baf3d7a46d3208683f",
    "workshop4_autoencoders/checkpoints/ae_best.pt": "91bfadc0949319edcb1eb60c8980235d26c7eb5222d9e628c41b59c38616492c",
    "workshop4_autoencoders/checkpoints/vae_best.pt": "3c916aeb8da593cba3f8adb21884e82c4d2d60d7df1d73e93b940f8e79983187",
    "workshop4_autoencoders/checkpoints/ae_latent_64.pt": "011f181f9c9caead994abd7c29d54aa53160287a230dbed4796ac57a3ad25b58",
    "checkpoints/part_a/model_e4_wide_128_64_seed42.pt": "3ed12f21a70ca642bf443d5e22aaf7656ff20d87e1d42cefaad79c816614a076",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


notebook = nbformat.read(NOTEBOOK, as_version=4)
markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
image_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.output_type in {"display_data", "execute_result"}
    and "image/png" in output.get("data", {})
]
unexecuted = [index for index, cell in enumerate(code_cells) if cell.execution_count is None]
execution_counts = [cell.execution_count for cell in code_cells]

source_text = "\n".join(cell.source for cell in notebook.cells)
rendered_text_parts = []
for cell in code_cells:
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            rendered_text_parts.append(output.get("text", ""))
        elif output.output_type in {"display_data", "execute_result"}:
            rendered_text_parts.extend(
                str(output.get("data", {}).get(key, ""))
                for key in ("text/plain", "text/html")
            )
rendered_text = "\n".join(rendered_text_parts)

absolute_windows_paths = re.findall(r"[A-Za-z]:[\\/]", source_text + "\n" + rendered_text)
training_calls = [
    pattern
    for pattern in (
        r"\btrain_autoencoder\s*\(",
        r"\btrain_vae\s*\(",
        r"\.backward\s*\(",
        r"optimizer\.step\s*\(",
        r"\.fit_transform\s*\(",
        r"\bTSNE\s*\(",
        r"\bUMAP\s*\(",
    )
    if re.search(pattern, source_text)
]

artifact_hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED_HASHES}

decoded_images = [base64.b64decode(output["data"]["image/png"], validate=True) for output in image_outputs]
embedded_image_hashes = {hashlib.sha256(data).hexdigest() for data in decoded_images}
saved_figure_hashes = {
    sha256(path)
    for path in (ROOT / "workshop4_autoencoders" / "results" / "figures").rglob("*.png")
}

latent = np.load(ROOT / "workshop4_autoencoders/results/latent_representations.npz")
tsne = np.load(ROOT / "workshop4_autoencoders/results/latent_tsne_embeddings.npz")
umap = np.load(ROOT / "workshop4_autoencoders/results/latent_umap_embeddings.npz")
expected_latent_shapes = {
    "AE z=128": (440, 128),
    "AE z=64": (440, 64),
    "VAE z=64 beta=0.5": (440, 64),
    "MLP z=64": (440, 64),
}
expected_embedding_shapes = {key: (440, 2) for key in expected_latent_shapes}

result_tokens = {
    "AE validation MSE": "0.452771",
    "VAE validation MSE": "1.012741",
    "AE test reconstruction MSE": "0.399277",
    "AE anomaly ROC-AUC": "0.631728",
    "VAE sale-condition silhouette": "0.516616",
    "AE64 t-SNE trustworthiness": "0.859103",
}

readme_text = README.read_text(encoding="utf-8")

assert len(markdown_cells) == 17, len(markdown_cells)
assert len(code_cells) == 15, len(code_cells)
assert not error_outputs, f"Notebook contains {len(error_outputs)} error outputs"
assert not unexecuted, f"Unexecuted code cells: {unexecuted}"
assert execution_counts == list(range(1, 16)), execution_counts
assert len(image_outputs) == 25, len(image_outputs)
assert all(data.startswith(b"\x89PNG\r\n\x1a\n") for data in decoded_images)
assert embedded_image_hashes <= saved_figure_hashes, "An embedded PNG differs from its saved source figure"
assert all(output.get("metadata", {}).get("image/png", {}).get("alt") for output in image_outputs)
assert not absolute_windows_paths, f"Absolute Windows paths found: {absolute_windows_paths}"
assert not training_calls, f"Training or reducer fitting calls found: {training_calls}"
assert artifact_hashes == EXPECTED_HASHES, "A protected split/checkpoint hash changed"
assert {key: latent[key].shape for key in expected_latent_shapes} == expected_latent_shapes
assert {key: tsne[key].shape for key in expected_embedding_shapes} == expected_embedding_shapes
assert {key: umap[key].shape for key in expected_embedding_shapes} == expected_embedding_shapes
for label, token in result_tokens.items():
    assert token in rendered_text, f"Rendered result missing: {label} ({token})"
for artifact_name in (
    "ae_comparison.csv",
    "vae_comparison.csv",
    "reconstruction_metrics.csv",
    "anomaly_metrics.csv",
    "latent_metrics.csv",
    "latent_representations.npz",
    "latent_tsne_embeddings.npz",
    "latent_umap_embeddings.npz",
    "data_metadata.json",
    "anomaly_protocol_metadata.json",
    "latent_embedding_metadata.json",
):
    assert artifact_name in source_text, f"Notebook does not load source artifact: {artifact_name}"
for required in (
    "notebooks/workshop4_autoencoders.ipynb",
    "jupyter.exe notebook",
    "nbconvert --to notebook --execute --inplace",
    "validate_workshop4_notebook.py",
    "does not retrain models",
):
    assert required in readme_text, f"README instruction missing: {required}"

print(f"Notebook: {NOTEBOOK}")
print(f"Markdown cells: {len(markdown_cells)}")
print(f"Code cells: {len(code_cells)}")
print(f"Embedded PNG figures: {len(image_outputs)}")
print("Error outputs: 0")
print("Unexecuted code cells: 0")
print(f"Execution counts: {execution_counts[0]}-{execution_counts[-1]}")
print("Embedded PNG signatures and source-file hashes verified")
print("Embedded PNG alternative-text metadata verified")
print("Absolute Windows paths: 0")
print("Training/reducer fitting calls: 0")
print(f"Protected artifact hashes verified: {len(artifact_hashes)}")
print("Saved NPZ array shapes verified")
print(f"Rendered source-result values verified: {len(result_tokens)}")
print("README open/reproduction instructions verified")
