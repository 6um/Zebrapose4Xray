import os
import json
import argparse
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_history(save_dir: str) -> pd.DataFrame:
    history_csv = os.path.join(save_dir, "history.csv")
    history_json = os.path.join(save_dir, "history.json")

    if os.path.exists(history_csv):
        df = pd.read_csv(history_csv)
        print(f"[INFO] Loaded history from {history_csv}")
        return df

    if os.path.exists(history_json):
        with open(history_json, "r", encoding="utf-8") as f:
            history = json.load(f)
        df = pd.DataFrame(history)
        print(f"[INFO] Loaded history from {history_json}")
        return df

    raise FileNotFoundError(
        f"Neither history.csv nor history.json found in {save_dir}"
    )


def load_best_info(save_dir: str) -> Dict:
    best_info_path = os.path.join(save_dir, "best_metrics.json")
    if os.path.exists(best_info_path):
        with open(best_info_path, "r", encoding="utf-8") as f:
            best_info = json.load(f)
        print(f"[INFO] Loaded best metrics from {best_info_path}")
        return best_info
    print("[WARN] best_metrics.json not found, will infer best epoch from val_loss.")
    return {}


def safe_get(series: pd.Series, default=np.nan):
    try:
        return series.item()
    except Exception:
        return default


def summarize_results(df: pd.DataFrame, best_info: Dict) -> Dict:
    if "val_loss" not in df.columns:
        raise KeyError("Column 'val_loss' not found in history data.")

    if best_info and "metrics_at_best" in best_info:
        best_epoch = int(best_info["best_epoch"])
        best_row = df[df["epoch"] == best_epoch].iloc[0]
    else:
        best_idx = df["val_loss"].idxmin()
        best_row = df.loc[best_idx]
        best_epoch = int(best_row["epoch"])

    last_row = df.iloc[-1]

    summary = {
        "num_epochs": int(len(df)),
        "best_epoch": best_epoch,

        "best_val_loss": float(best_row.get("val_loss", np.nan)),
        "best_val_mask_loss": float(best_row.get("val_mask_loss", np.nan)),
        "best_val_code_loss": float(best_row.get("val_code_loss", np.nan)),

        "best_val_mask_iou": float(best_row.get("val_mask_iou", np.nan)),
        "best_val_mask_precision": float(best_row.get("val_mask_precision", np.nan)),
        "best_val_mask_recall": float(best_row.get("val_mask_recall", np.nan)),
        "best_val_mask_f1": float(best_row.get("val_mask_f1", np.nan)),
        "best_val_mask_acc": float(best_row.get("val_mask_acc", np.nan)),

        "best_val_code_acc": float(best_row.get("val_code_acc", np.nan)),
        "best_val_fg_code_acc": float(best_row.get("val_fg_code_acc", np.nan)),
        "best_val_bg_code_acc": float(best_row.get("val_bg_code_acc", np.nan)),
        "best_val_code_bitwise_precision_fg": float(best_row.get("val_code_bitwise_precision_fg", np.nan)),
        "best_val_code_bitwise_recall_fg": float(best_row.get("val_code_bitwise_recall_fg", np.nan)),
        "best_val_code_bitwise_f1_fg": float(best_row.get("val_code_bitwise_f1_fg", np.nan)),
        "best_val_code_fg_all_correct": float(best_row.get("val_code_fg_all_correct", np.nan)),

        "last_epoch": int(last_row.get("epoch", len(df))),
        "last_train_loss": float(last_row.get("train_loss", np.nan)),
        "last_train_mask_loss": float(last_row.get("train_mask_loss", np.nan)),
        "last_train_code_loss": float(last_row.get("train_code_loss", np.nan)),
        "last_val_loss": float(last_row.get("val_loss", np.nan)),
        "last_val_mask_loss": float(last_row.get("val_mask_loss", np.nan)),
        "last_val_code_loss": float(last_row.get("val_code_loss", np.nan)),
        "last_val_mask_iou": float(last_row.get("val_mask_iou", np.nan)),
        "last_val_code_acc": float(last_row.get("val_code_acc", np.nan)),
        "last_val_fg_code_acc": float(last_row.get("val_fg_code_acc", np.nan)),
        "last_val_code_fg_all_correct": float(last_row.get("val_code_fg_all_correct", np.nan)),
    }

    return summary


def print_report_ready_summary(summary: Dict):
    print("\n" + "=" * 70)
    print("REPORT-READY SUMMARY")
    print("=" * 70)

    print(f"Total epochs: {summary['num_epochs']}")
    print(f"Best epoch: {summary['best_epoch']}")

    print("\nBest validation results:")
    print(f"  Val Loss                    : {summary['best_val_loss']:.4f}")
    print(f"  Val Mask Loss               : {summary['best_val_mask_loss']:.4f}")
    print(f"  Val Code Loss               : {summary['best_val_code_loss']:.4f}")
    print(f"  Val Mask IoU                : {summary['best_val_mask_iou']:.4f}")
    print(f"  Val Mask Precision          : {summary['best_val_mask_precision']:.4f}")
    print(f"  Val Mask Recall             : {summary['best_val_mask_recall']:.4f}")
    print(f"  Val Mask F1                 : {summary['best_val_mask_f1']:.4f}")
    print(f"  Val Mask Accuracy           : {summary['best_val_mask_acc']:.4f}")
    print(f"  Val Code Accuracy           : {summary['best_val_code_acc']:.4f}")
    print(f"  Val FG Code Accuracy        : {summary['best_val_fg_code_acc']:.4f}")
    print(f"  Val BG Code Accuracy        : {summary['best_val_bg_code_acc']:.4f}")
    print(f"  Val Code Precision (FG)     : {summary['best_val_code_bitwise_precision_fg']:.4f}")
    print(f"  Val Code Recall (FG)        : {summary['best_val_code_bitwise_recall_fg']:.4f}")
    print(f"  Val Code F1 (FG)            : {summary['best_val_code_bitwise_f1_fg']:.4f}")
    print(f"  Val Full 10-bit Correct(FG) : {summary['best_val_code_fg_all_correct']:.4f}")

    print("\nLast epoch results:")
    print(f"  Epoch                       : {summary['last_epoch']}")
    print(f"  Train Loss                  : {summary['last_train_loss']:.4f}")
    print(f"  Train Mask Loss             : {summary['last_train_mask_loss']:.4f}")
    print(f"  Train Code Loss             : {summary['last_train_code_loss']:.4f}")
    print(f"  Val Loss                    : {summary['last_val_loss']:.4f}")
    print(f"  Val Mask Loss               : {summary['last_val_mask_loss']:.4f}")
    print(f"  Val Code Loss               : {summary['last_val_code_loss']:.4f}")
    print(f"  Val Mask IoU                : {summary['last_val_mask_iou']:.4f}")
    print(f"  Val Code Accuracy           : {summary['last_val_code_acc']:.4f}")
    print(f"  Val FG Code Accuracy        : {summary['last_val_fg_code_acc']:.4f}")
    print(f"  Val Full 10-bit Correct(FG) : {summary['last_val_code_fg_all_correct']:.4f}")

    print("\nRecommended metrics to report:")
    print("  - Val Mask IoU")
    print("  - Val Mask F1")
    print("  - Val FG Code Accuracy")
    print("  - Val Code F1 (FG)")
    print("  - Val Full 10-bit Correctness on FG")


def build_tables(summary: Dict) -> Dict[str, pd.DataFrame]:
    main_table = pd.DataFrame([
        ["Mask IoU", summary["best_val_mask_iou"]],
        ["Mask F1", summary["best_val_mask_f1"]],
        ["Mask Precision", summary["best_val_mask_precision"]],
        ["Mask Recall", summary["best_val_mask_recall"]],
        ["Code Accuracy (all pixels)", summary["best_val_code_acc"]],
        ["FG Code Accuracy", summary["best_val_fg_code_acc"]],
        ["BG Code Accuracy", summary["best_val_bg_code_acc"]],
        ["Code Bitwise Precision (FG)", summary["best_val_code_bitwise_precision_fg"]],
        ["Code Bitwise Recall (FG)", summary["best_val_code_bitwise_recall_fg"]],
        ["Code Bitwise F1 (FG)", summary["best_val_code_bitwise_f1_fg"]],
        ["Full 10-bit Correctness (FG)", summary["best_val_code_fg_all_correct"]],
    ], columns=["Metric", "Value"])

    loss_table = pd.DataFrame([
        ["Best Val Loss", summary["best_val_loss"]],
        ["Best Val Mask Loss", summary["best_val_mask_loss"]],
        ["Best Val Code Loss", summary["best_val_code_loss"]],
        ["Last Train Loss", summary["last_train_loss"]],
        ["Last Train Mask Loss", summary["last_train_mask_loss"]],
        ["Last Train Code Loss", summary["last_train_code_loss"]],
        ["Last Val Loss", summary["last_val_loss"]],
        ["Last Val Mask Loss", summary["last_val_mask_loss"]],
        ["Last Val Code Loss", summary["last_val_code_loss"]],
    ], columns=["Metric", "Value"])

    meta_table = pd.DataFrame([
        ["Number of Epochs", summary["num_epochs"]],
        ["Best Epoch", summary["best_epoch"]],
        ["Last Epoch", summary["last_epoch"]],
    ], columns=["Item", "Value"])

    return {
        "main_results": main_table,
        "loss_results": loss_table,
        "training_meta": meta_table,
    }


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for name, table in tables.items():
        csv_path = os.path.join(out_dir, f"{name}.csv")
        txt_path = os.path.join(out_dir, f"{name}.txt")

        table.to_csv(csv_path, index=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table.to_string(index=False))

        print(f"[INFO] Saved table: {csv_path}")
        print(f"[INFO] Saved table: {txt_path}")


def save_summary_text(summary: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_summary.txt")

    lines = [
        "Training Result Summary",
        "=" * 50,
        f"Total epochs: {summary['num_epochs']}",
        f"Best epoch: {summary['best_epoch']}",
        "",
        "Best validation metrics:",
        f"Val Loss: {summary['best_val_loss']:.4f}",
        f"Val Mask Loss: {summary['best_val_mask_loss']:.4f}",
        f"Val Code Loss: {summary['best_val_code_loss']:.4f}",
        f"Val Mask IoU: {summary['best_val_mask_iou']:.4f}",
        f"Val Mask Precision: {summary['best_val_mask_precision']:.4f}",
        f"Val Mask Recall: {summary['best_val_mask_recall']:.4f}",
        f"Val Mask F1: {summary['best_val_mask_f1']:.4f}",
        f"Val Mask Accuracy: {summary['best_val_mask_acc']:.4f}",
        f"Val Code Accuracy: {summary['best_val_code_acc']:.4f}",
        f"Val FG Code Accuracy: {summary['best_val_fg_code_acc']:.4f}",
        f"Val BG Code Accuracy: {summary['best_val_bg_code_acc']:.4f}",
        f"Val Code Bitwise Precision (FG): {summary['best_val_code_bitwise_precision_fg']:.4f}",
        f"Val Code Bitwise Recall (FG): {summary['best_val_code_bitwise_recall_fg']:.4f}",
        f"Val Code Bitwise F1 (FG): {summary['best_val_code_bitwise_f1_fg']:.4f}",
        f"Val Full 10-bit Correctness (FG): {summary['best_val_code_fg_all_correct']:.4f}",
        "",
        "Recommended report metrics:",
        "- Mask IoU",
        "- Mask F1",
        "- FG Code Accuracy",
        "- Code Bitwise F1 (FG)",
        "- Full 10-bit Correctness (FG)",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Saved summary text: {path}")


def plot_curve(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, ylabel: str, save_path: str):
    plt.figure(figsize=(8, 5))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot: {save_path}")


def generate_plots(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    plot_curve(
        df,
        "epoch",
        ["train_loss", "val_loss"],
        "Training and Validation Loss",
        "Loss",
        os.path.join(out_dir, "loss_curve.png"),
    )

    plot_curve(
        df,
        "epoch",
        ["train_mask_loss", "val_mask_loss", "train_code_loss", "val_code_loss"],
        "Mask and Code Loss Curves",
        "Loss",
        os.path.join(out_dir, "mask_code_loss_curve.png"),
    )

    plot_curve(
        df,
        "epoch",
        ["val_mask_iou", "val_mask_f1", "val_mask_precision", "val_mask_recall"],
        "Mask Metrics",
        "Score",
        os.path.join(out_dir, "mask_metrics_curve.png"),
    )

    plot_curve(
        df,
        "epoch",
        ["val_code_acc", "val_fg_code_acc", "val_bg_code_acc"],
        "Code Accuracy Metrics",
        "Score",
        os.path.join(out_dir, "code_accuracy_curve.png"),
    )

    plot_curve(
        df,
        "epoch",
        [
            "val_code_bitwise_precision_fg",
            "val_code_bitwise_recall_fg",
            "val_code_bitwise_f1_fg",
            "val_code_fg_all_correct",
        ],
        "Foreground Code Quality Metrics",
        "Score",
        os.path.join(out_dir, "fg_code_metrics_curve.png"),
    )

    if "lr" in df.columns:
        plot_curve(
            df,
            "epoch",
            ["lr"],
            "Learning Rate",
            "LR",
            os.path.join(out_dir, "learning_rate_curve.png"),
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze saved training history and generate report-ready outputs.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing history.csv/json and best_metrics.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for tables and plots")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(save_dir, "analysis")

    os.makedirs(out_dir, exist_ok=True)

    df = load_history(save_dir)
    best_info = load_best_info(save_dir)

    summary = summarize_results(df, best_info)
    print_report_ready_summary(summary)

    tables = build_tables(summary)
    save_tables(tables, out_dir)
    save_summary_text(summary, out_dir)
    generate_plots(df, out_dir)

    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved summary json: {summary_json_path}")

    print("\n[INFO] Analysis finished.")
    print(f"[INFO] Output directory: {out_dir}")


if __name__ == "__main__":
    main()

'''
python analyze_training_results.py --save_dir ./checkpoints --out_dir ./checkpoints/analysis

'''