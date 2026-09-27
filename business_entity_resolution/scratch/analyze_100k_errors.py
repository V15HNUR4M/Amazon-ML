import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load 100k report
with open("output/turn8_5_100k/turn7_validation_report.json", "r") as f:
    r100k = json.load(f)

# Load 5k report
with open("output/turn7_validation_report.json", "r") as f:
    r5k = json.load(f)

print("="*80)
print("ERROR DECOMPOSITION: 5k vs 100k")
print("="*80)

# True matches in val
n_val_5k = r5k["error_analysis"]["total_val_entities"]
n_val_100k = r100k["error_analysis"]["total_val_entities"]

trues_5k = r5k["error_analysis"]["total_true_matches"]
trues_100k = r100k["error_analysis"]["total_true_matches"]

tp_5k = r5k["error_analysis"]["true_positives"]
tp_100k = r100k["error_analysis"]["true_positives"]

fp_5k = r5k["error_analysis"]["false_positives"]
fp_100k = r100k["error_analysis"]["false_positives"]

fn_block_5k = r5k["error_analysis"]["blocking_misses_count"]
fn_block_100k = r100k["error_analysis"]["blocking_misses_count"]

fn_class_5k = r5k["error_analysis"]["classifier_misses_count"]
fn_class_100k = r100k["error_analysis"]["classifier_misses_count"]

print(f"{'Metric':<35} | {'5k Scale (1k Val)':<20} | {'100k Scale (20k Val)':<20} | {'Ratio (100k / 5k)':<15}")
print("-" * 95)
print(f"{'Val Entities':<35} | {n_val_5k:<20} | {n_val_100k:<20} | {n_val_100k / n_val_5k:.1f}x")
print(f"{'True Matches':<35} | {trues_5k:<20} | {trues_100k:<20} | {trues_100k / trues_5k:.2f}x")
print(f"{'True Matches / Entity':<35} | {trues_5k / n_val_5k:<20.3f} | {trues_100k / n_val_100k:<20.3f} | {(trues_100k / n_val_100k) / (trues_5k / n_val_5k):.3f}x")
print(f"{'True Positives':<35} | {tp_5k:<20} | {tp_100k:<20} | {tp_100k / tp_5k:.2f}x")
print(f"{'TP Rate (% of trues)':<35} | {tp_5k / trues_5k * 100:<20.2f}% | {tp_100k / trues_100k * 100:<20.2f}% | {(tp_100k / trues_100k) / (tp_5k / trues_5k):.3f}x")
print(f"{'Blocking Misses':<35} | {fn_block_5k:<20} | {fn_block_100k:<20} | {fn_block_100k / fn_block_5k:.1f}x")
print(f"{'Blocking Miss Rate':<35} | {fn_block_5k / trues_5k * 100:<20.2f}% | {fn_block_100k / trues_100k * 100:<20.2f}% | {(fn_block_100k / trues_100k) / (fn_block_5k / trues_5k):.2f}x")
print(f"{'Classifier Misses':<35} | {fn_class_5k:<20} | {fn_class_100k:<20} | {fn_class_100k / fn_class_5k:.1f}x")
print(f"{'Classifier Miss Rate':<35} | {fn_class_5k / trues_5k * 100:<20.2f}% | {fn_class_100k / trues_100k * 100:<20.2f}% | {(fn_class_100k / trues_100k) / (fn_class_5k / trues_5k):.2f}x")
print(f"{'False Positives (Total)':<35} | {fp_5k:<20} | {fp_100k:<20} | {fp_100k / fp_5k:.1f}x")
print(f"{'False Positives / Val Entity':<35} | {fp_5k / n_val_5k:<20.4f} | {fp_100k / n_val_100k:<20.4f} | {(fp_100k / n_val_100k) / (fp_5k / n_val_5k):.2f}x")

print("\n--- Impact on Macro F0.5 Components ---")
p5 = r5k["validation_metrics"]["mean_entity_precision"]
r5 = r5k["validation_metrics"]["mean_entity_recall"]
f5 = r5k["validation_metrics"]["macro_f05"]

p100 = r100k["validation_metrics"]["mean_entity_precision"]
r100 = r100k["validation_metrics"]["mean_entity_recall"]
f100 = r100k["validation_metrics"]["macro_f05"]

print(f"5k:   Precision = {p5:.4f}, Recall = {r5:.4f} -> Macro F0.5 = {f5:.4f}")
print(f"100k: Precision = {p100:.4f}, Recall = {r100:.4f} -> Macro F0.5 = {f100:.4f}")
print(f"Delta: Precision = {p100 - p5:+.4f}, Recall = {r100 - r5:+.4f} -> Delta F0.5 = {f100 - f5:+.4f}")

# Sensitivity analysis: How much does Precision drop contribute vs Recall drop?
# F0.5 with 5k precision but 100k recall
f_p5_r100 = (1.25 * p5 * r100) / (0.25 * p5 + r100)
# F0.5 with 100k precision but 5k recall
f_p100_r5 = (1.25 * p100 * r5) / (0.25 * p100 + r5)

print("\nF0.5 Sensitivity Breakdown:")
print(f"  Holding 5k Precision (0.9893) constant, 100k Recall (0.9428) gives F0.5 = {f_p5_r100:.4f} (Drop: {f_p5_r100 - f5:+.4f})")
print(f"  Holding 5k Recall (0.9651) constant, 100k Precision (0.9812) gives F0.5 = {f_p100_r5:.4f} (Drop: {f_p100_r5 - f5:+.4f})")
print(f"  Recall drop accounts for: {abs(f_p5_r100 - f5) / (abs(f_p5_r100 - f5) + abs(f_p100_r5 - f5)) * 100:.1f}% of total drop")
print(f"  Precision drop accounts for: {abs(f_p100_r5 - f5) / (abs(f_p5_r100 - f5) + abs(f_p100_r5 - f5)) * 100:.1f}% of total drop")

