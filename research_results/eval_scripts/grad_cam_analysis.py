"""
grad_cam_analysis.py — Grad-CAM Visualization for PoultryScan AI

Generates Grad-CAM saliency overlays for the active 5-class MobileNetV2 model.

Architecture:
    Input -> DataAugmentation -> MobileNetV2 -> GlobalAveragePooling2D -> Dropout -> Dense(5, softmax)
    Grad-CAM targets the last 4-D spatial feature map inside MobileNetV2
    (before GlobalAveragePooling), found programmatically.

Limitations (IMPORTANT):
    Grad-CAM is a QUALITATIVE explanation aid only.
    Highlighted regions show where the gradient signal is strongest.
    This does NOT prove causality or guarantee prediction correctness.

Usage:
    python grad_cam_analysis.py --model model/poultry_disease_model.keras \\
        --test-dir test_dataset --output research_results/grad_cam_examples \\
        --samples-per-class 3
"""

import argparse
import json
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_class_indices(model_path):
    """Load class mapping from class_indices.json next to the model file."""
    ci = os.path.join(os.path.dirname(model_path) or ".", "class_indices.json")
    if os.path.exists(ci):
        with open(ci) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data[str(i)] for i in range(len(data))]
    return ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]


def preprocess(image_path):
    """RGB -> 224x224 bilinear -> /255.0 -> batch dim."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return img, np.expand_dims(arr, 0)


def find_last_spatial_layer_name(base_model):
    """
    Return the name of the last spatial (4-D) feature layer in MobileNetV2.

    In Keras 3, individual layer objects do NOT expose `output_shape` unless
    accessed through the functional model's computed call graph. Instead we
    probe a priority list of known MobileNetV2 terminal layer names and fall
    back to the last non-InputLayer in the sub-model.

    MobileNetV2 terminal spatial layers (in preference order):
        out_relu -> Conv_1_bn -> Conv_1 -> block_16_project_BN
    """
    priority = ["out_relu", "Conv_1_bn", "Conv_1", "block_16_project_BN",
                "block_16_project"]
    layer_names = {l.name for l in base_model.layers}

    for name in priority:
        if name in layer_names:
            return name

    # Fallback: last non-InputLayer
    last_name = None
    for layer in base_model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            last_name = layer.name
    if last_name is None:
        raise RuntimeError(
            "Cannot find any usable layer in the MobileNetV2 sub-model."
        )
    return last_name



def make_gradcam_heatmap(input_tensor, model, class_index):
    """
    Compute Grad-CAM heatmap using GradientTape (Keras 3 compatible).

    In Keras 3, sub-model layers cannot be rewired with the outer model's inputs.
    Solution: build a feature extractor from base_model.input -> target_layer.output,
    then manually forward the preprocessed tensor through it and the remaining head.
    """
    aug_layer = None
    base_model = None
    gap_layer = None
    dropout_layer = None
    dense_layer = None

    for layer in model.layers:
        lname = layer.name.lower()
        if "augment" in lname or isinstance(layer, tf.keras.Sequential):
            aug_layer = layer
        elif "mobilenetv2" in lname:
            base_model = layer
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        elif isinstance(layer, tf.keras.layers.Dropout):
            dropout_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer

    if base_model is None:
        raise RuntimeError("MobileNetV2 sub-model not found in model graph.")

    target_layer_name = find_last_spatial_layer_name(base_model)

    # Build feature extractor using base_model's own input
    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(target_layer_name).output
    )

    # Run augmentation (if any) outside the tape
    x = tf.cast(input_tensor, tf.float32)
    if aug_layer is not None:
        x = aug_layer(x, training=False)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(x, training=False)
        tape.watch(conv_outputs)
        z = gap_layer(conv_outputs, training=False)
        if dropout_layer is not None:
            z = dropout_layer(z, training=False)
        predictions = dense_layer(z, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), target_layer_name



def overlay_heatmap(original_pil, heatmap, alpha=0.45):
    """Resize heatmap to image size and blend with original image."""
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize(original_pil.size, Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_resized)
    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(heatmap_np / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    orig_np = np.array(original_pil)
    overlay = (alpha * heatmap_colored + (1 - alpha) * orig_np).astype(np.uint8)
    return overlay, heatmap_colored


def save_gradcam_figure(orig_pil, heatmap_np, overlay_np, info, out_path):
    """Save a 3-panel figure: original | heatmap | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(orig_pil)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap_np, cmap="jet")
    axes[1].set_title(f"Grad-CAM Heatmap\n(layer: {info['target_layer']})", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(overlay_np)
    verdict = "Correct" if info["correct"] else "Wrong"
    color = "green" if info["correct"] else "red"
    axes[2].set_title(
        f"Overlay | Pred: {info['pred_class']} ({info['confidence']:.1f}%)\n"
        f"GT: {info['gt_class']} — {verdict}",
        fontsize=10, color=color
    )
    axes[2].axis("off")

    fig.suptitle(
        "Grad-CAM Qualitative Explanation — PoultryScan AI MobileNetV2\n"
        "IMPORTANT: Highlighted regions show gradient magnitude only. "
        "Not proof of causality or correctness.",
        fontsize=9, color="#555555", style="italic"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_gradcam(model_path, test_dir, output_dir, samples_per_class, seed):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 65)
    print("POULTRYSCAN AI - GRAD-CAM QUALITATIVE ANALYSIS")
    print("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    class_names = load_class_indices(model_path)
    print(f"Classes ({len(class_names)}): {class_names}")

    os.makedirs(output_dir, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    base_model_name = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            base_model_name = layer.name
            break

    manifest = []
    total_processed = 0

    for cls_name in class_names:
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  [SKIP] Class folder not found: {cls_dir}")
            continue

        images = [f for f in os.listdir(cls_dir)
                  if os.path.splitext(f)[1].lower() in valid_exts]
        if not images:
            print(f"  [SKIP] No images in {cls_dir}")
            continue

        selected = random.sample(images, min(samples_per_class, len(images)))
        cls_index = class_names.index(cls_name)
        print(f"\nClass: {cls_name} (index {cls_index}) — processing {len(selected)} images")

        for img_file in selected:
            img_path = os.path.join(cls_dir, img_file)
            try:
                orig_pil, input_tensor = preprocess(img_path)
                probs = model.predict(input_tensor, verbose=0)[0]
                pred_index = int(np.argmax(probs))
                pred_class = class_names[pred_index]
                confidence = float(probs[pred_index]) * 100.0
                correct = (pred_index == cls_index)

                heatmap_np, target_layer = make_gradcam_heatmap(
                    input_tensor, model, pred_index
                )
                overlay_np, _ = overlay_heatmap(orig_pil, heatmap_np)

                tag = "correct" if correct else "wrong"
                safe_cls = cls_name.replace(" ", "_")
                safe_pred = pred_class.replace(" ", "_")
                base_name = os.path.splitext(img_file)[0]
                fig_name = f"{safe_cls}_gt__{safe_pred}_pred__{tag}__{base_name}.png"
                fig_path = os.path.join(output_dir, fig_name)

                info = {
                    "gt_class": cls_name,
                    "pred_class": pred_class,
                    "confidence": round(confidence, 2),
                    "correct": correct,
                    "target_layer": target_layer,
                    "source_image": img_path,
                    "output_figure": fig_path,
                    "all_probabilities": {
                        class_names[i]: round(float(probs[i]) * 100, 2)
                        for i in range(len(class_names))
                    }
                }

                save_gradcam_figure(orig_pil, heatmap_np, overlay_np, info, fig_path)
                manifest.append(info)
                total_processed += 1

                status = "CORRECT" if correct else "WRONG"
                print(f"  [{status}] {img_file} -> {pred_class} ({confidence:.1f}%) "
                      f"| layer: {target_layer}")

            except Exception as exc:
                print(f"  [ERROR] {img_file}: {exc}")

    manifest_path = os.path.join(output_dir, "gradcam_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "model": model_path,
            "base_mobilenetv2_layer": base_model_name,
            "grad_cam_method": "GradCAM via GradientTape on last 4-D spatial layer of MobileNetV2",
            "target_class": "PREDICTED class (argmax of softmax), not ground-truth",
            "grad_cam_disclaimer": (
                "Grad-CAM is a qualitative explanation aid. Highlighted regions show "
                "where the gradient of the predicted class score w.r.t. the last "
                "MobileNetV2 spatial feature map is largest. This does NOT prove "
                "causality or prediction correctness."
            ),
            "seed": seed,
            "samples_per_class": samples_per_class,
            "total_figures": total_processed,
            "figures": manifest
        }, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"Grad-CAM complete: {total_processed} figures saved to '{output_dir}'")
    print(f"Manifest: {manifest_path}")
    print("=" * 65)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM qualitative analysis for PoultryScan AI."
    )
    parser.add_argument("--model", default="model/poultry_disease_model.keras")
    parser.add_argument("--test-dir", default="test_dataset")
    parser.add_argument("--output", default="research_results/grad_cam_examples")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_gradcam(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
