"""
Joint Exposure-Demand v2 END-TO-END (SAME IO, LOCAL CSV)
---------------------------------------------------------
Goal:
  - Keep notebook input style unchanged.
  - Train exposure and demand in one differentiable model.
  - Exposure branch predicts total -> buy_box/in_stock hierarchy.
  - Demand branch consumes the predicted exposure hats directly in-memory.
  - Demand loss can backpropagate into exposure branch (true end-to-end).
  - Still saves exposure_hat_for_demand.csv for compatibility.

Default design:
  - Pretrain exposure a few epochs (optional but strongly recommended).
  - Then joint train:
        L = L_exposure + lambda_demand * L_demand + lambda_noexp * L_no_exposure_no_demand
  - Scheduled teacher forcing for demand exposure input:
        exposure_for_demand = tf_rate * true_exposure_log + (1-tf_rate) * pred_exposure_log
    with default tf_rate decaying from 0.50 to 0.00.
  - If detach_exposure_for_demand=False, demand gradients flow into exposure predictions.

Run:
  %run -i joint_exposure_demand_v2_END2END_SAMEIO_LOCALCSV.py

  joint_result_v2 = run_joint_exposure_demand_end2end_v2(
      data_raw1=data_raw1,
      scot_df=scot_df,
      n_asins=5000,
      exposure_history=13,
      demand_history=52,
      horizon=20,
      pretrain_exposure_epochs=8,
      joint_epochs=40,
      batch_size=64,
      patience=8,
  )
"""

import os
import runpy
import math
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# Utilities
# -------------------------------

def _find_script(filename):
    candidates = [Path(os.getcwd()) / filename, Path("/mnt/data") / filename]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Cannot find {filename} in cwd or /mnt/data")


def _to_device_batch(b, device):
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _to_device_batch(v, device)
        else:
            out[k] = v
    return out


def _simple_wape_summary(forecast_df, remove_oos_dp=True):
    df = forecast_df.copy()
    if remove_oos_dp and "scot_oos" in df.columns:
        df = df[df["scot_oos"].fillna(0).astype(float) == 0].copy()
    denom = df["fbi_demand"].abs().sum()
    denom = max(float(denom), 1e-9)
    out = {}
    for q in ["p50", "p70"]:
        pred = df[f"{q}_amxl"].astype(float)
        y = df["fbi_demand"].astype(float)
        err = pred - y
        out[f"{q}_amxl_wape"] = float(err.abs().sum() / denom)
        out[f"{q}_amxl_overbias"] = float(err.clip(lower=0).sum() / denom)
        out[f"{q}_amxl_underbias"] = float((-err.clip(upper=0)).sum() / denom)
        if f"{q}_scot" in df.columns:
            sp = df[f"{q}_scot"].astype(float)
            se = sp - y
            out[f"{q}_scot_wape"] = float(se.abs().sum() / denom)
            out[f"{q}_scot_overbias"] = float(se.clip(lower=0).sum() / denom)
            out[f"{q}_scot_underbias"] = float((-se.clip(upper=0)).sum() / denom)
    return out


def _merge_scot_if_available(forecast_df, scot_df):
    if scot_df is None:
        return forecast_df
    req = {"asin", "order_week", "forecast_qty_p50", "forecast_qty_p70"}
    scot = scot_df.copy()
    scot.columns = [c.strip() for c in scot.columns]
    if not req.issubset(set(scot.columns)):
        print("SCOT columns not matched; use historical mean baseline only.")
        return forecast_df
    scot["asin"] = scot["asin"].astype(str)
    scot["order_week"] = pd.to_datetime(scot["order_week"])
    keep = (scot[["asin", "order_week", "forecast_qty_p50", "forecast_qty_p70"]]
            .dropna(subset=["asin", "order_week"])
            .groupby(["asin", "order_week"], as_index=False)
            .agg(forecast_qty_p50=("forecast_qty_p50", "mean"),
                 forecast_qty_p70=("forecast_qty_p70", "mean")))
    out = forecast_df.merge(keep, on=["asin", "order_week"], how="left")
    out["p50_scot"] = out["forecast_qty_p50"].fillna(out.get("p50_scot", np.nan))
    out["p70_scot"] = out["forecast_qty_p70"].fillna(out.get("p70_scot", np.nan))
    return out.drop(columns=[c for c in ["forecast_qty_p50", "forecast_qty_p70"] if c in out.columns])


def _print_summary(summary):
    print("\n" + "=" * 100)
    print("JOINT EXPOSURE-DEMAND v2 END-TO-END SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("=" * 100 + "\n")


# -------------------------------
# Paired dataset
# -------------------------------

class PairedExposureDemandDataset(Dataset):
    """Pair existing ExposureDataset and DemandDataset by (asin, first target week).

    This keeps each task's original feature engineering, graph context, and future_context.
    """
    def __init__(self, exposure_ds, demand_ds):
        exp_map = {}
        for i in range(len(exposure_ds)):
            item = exposure_ds[i]
            key = (str(item["asin"]), str(item["target_week"][0])[:10])
            exp_map[key] = i

        dem_map = {}
        for j in range(len(demand_ds)):
            item = demand_ds[j]
            key = (str(item["asin"]), str(item["target_week"][0])[:10])
            dem_map[key] = j

        common = sorted(set(exp_map.keys()) & set(dem_map.keys()))
        if len(common) == 0:
            raise RuntimeError(
                "No paired samples found between exposure and demand datasets. "
                "Check histories/horizon/date alignment."
            )
        self.exposure_ds = exposure_ds
        self.demand_ds = demand_ds
        self.pairs = [(exp_map[k], dem_map[k], k) for k in common]
        print(f"Paired samples: {len(self.pairs):,} / exposure={len(exposure_ds):,} / demand={len(demand_ds):,}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ei, di, key = self.pairs[idx]
        return {"exp": self.exposure_ds[ei], "dem": self.demand_ds[di], "pair_key": key}


def paired_collate(batch):
    exp_tensor_keys = [
        "x", "future_context", "future_total_dph", "future_buy_box_dph",
        "future_instock_dph", "future_demand",
    ]
    dem_tensor_keys = [
        "x", "future_context", "y", "oos", "our_price", "pkg_volume",
        "future_instock", "future_total_dph", "future_buy_box_dph",
    ]
    out = {"exp": {}, "dem": {}}
    for k in exp_tensor_keys:
        out["exp"][k] = torch.stack([b["exp"][k] for b in batch], dim=0)
    for k in dem_tensor_keys:
        if k in batch[0]["dem"]:
            out["dem"][k] = torch.stack([b["dem"][k] for b in batch], dim=0)
    out["exp"]["asin"] = [b["exp"]["asin"] for b in batch]
    out["dem"]["asin"] = [b["dem"]["asin"] for b in batch]
    # target_week for demand: match demand.generate_forecast_df convention: [H][B]
    H = len(batch[0]["dem"]["target_week"])
    out["dem"]["target_week"] = [[b["dem"]["target_week"][h] for b in batch] for h in range(H)]
    out["exp"]["target_week"] = [[b["exp"]["target_week"][h] for b in batch] for h in range(H)]
    out["pair_key"] = [b["pair_key"] for b in batch]
    return out


# -------------------------------
# Joint model
# -------------------------------

class JointExposureDemandModel(nn.Module):
    def __init__(self, exposure_model, demand_model):
        super().__init__()
        self.exposure_model = exposure_model
        self.demand_model = demand_model

    @staticmethod
    def make_true_exposure_log(exp_batch):
        true_level = torch.stack([
            exp_batch["future_total_dph"],
            exp_batch["future_buy_box_dph"],
            exp_batch["future_instock_dph"],
        ], dim=-1).clamp(min=0.0)
        return torch.log1p(true_level)

    def forward(
        self,
        exp_batch,
        dem_batch,
        nZ_demand=8,
        teacher_forcing_rate=0.0,
        detach_exposure_for_demand=False,
        return_aux=True,
    ):
        exp_aux = self.exposure_model(
            exp_batch["x"],
            exp_batch["future_context"],
            return_aux=True,
        )
        pred_exp_log = exp_aux["log_hat"]  # [B,H,3], differentiable
        true_exp_log = self.make_true_exposure_log(exp_batch).to(pred_exp_log.device)

        if detach_exposure_for_demand:
            pred_for_dem = pred_exp_log.detach()
        else:
            pred_for_dem = pred_exp_log

        if teacher_forcing_rate > 0:
            r = float(max(0.0, min(1.0, teacher_forcing_rate)))
            exposure_input_log = r * true_exp_log + (1.0 - r) * pred_for_dem
        else:
            exposure_input_log = pred_for_dem

        fc_dem = dem_batch["future_context"].clone()
        if fc_dem.shape[-1] < 3:
            raise RuntimeError("Demand future_context must have the last 3 external exposure hat columns.")
        fc_dem[:, :, -3:] = exposure_input_log

        demand_preds, demand_z_reg, stock_log_hat = self.demand_model(
            dem_batch["x"], fc_dem, nZ=nZ_demand
        )
        if return_aux:
            return {
                "exp_aux": exp_aux,
                "pred_exp_log": pred_exp_log,
                "true_exp_log": true_exp_log,
                "fc_dem": fc_dem,
                "demand_preds": demand_preds,
                "demand_z_reg": demand_z_reg,
                "stock_log_hat": stock_log_hat,
            }
        return demand_preds


# -------------------------------
# Losses
# -------------------------------

def demand_loss_from_preds(dem_ns, y, preds, z_reg, beta_tail=0.5, lambda_under=0.15, lambda_z_reg=1.0):
    nZ = len(preds)
    nll = sum(dem_ns["tail_weighted_negbin_nll"](y, mu, alpha, beta_tail=beta_tail)
              for mu, alpha in preds) / max(nZ, 1)
    mu_mean = torch.stack([mu for mu, _ in preds], dim=1).mean(dim=1)
    under = dem_ns["active_underforecast_loss"](y, mu_mean, log_scale=True)
    return nll + lambda_under * under + lambda_z_reg * z_reg, {"nll": nll.detach(), "under": under.detach()}


def no_exposure_no_demand_loss(pred_exp_log, demand_mu_mean, y, threshold_log=0.15, weight_true_zero_only=True):
    """Soft consistency: when predicted exposure is near zero, demand mu should be low.

    This is intentionally small. It should reduce overbias in low-exposure regions without
    forcing high-exposure samples to be active.
    """
    exposure_signal = torch.maximum(pred_exp_log[..., 1], pred_exp_log[..., 2])
    low_exp = torch.sigmoid((float(threshold_log) - exposure_signal) / 0.10)
    if weight_true_zero_only:
        zero_y = (y <= 0).float()
        w = low_exp * zero_y
    else:
        w = low_exp
    if w.sum() <= 0:
        return y.new_tensor(0.0)
    return (w * torch.log1p(demand_mu_mean.clamp(min=0.0))).sum() / w.sum().clamp(min=1.0)


# -------------------------------
# Train / predict
# -------------------------------

def train_joint_end2end(
    joint_model,
    exp_ns,
    dem_ns,
    tr_ld,
    va_ld,
    device,
    joint_epochs=40,
    lr=5e-4,
    patience=8,
    lambda_exposure=1.0,
    lambda_demand=1.0,
    lambda_noexp=0.03,
    beta_tail=0.5,
    lambda_under=0.15,
    lambda_z_reg=1.0,
    nZ_demand=6,
    teacher_forcing_start=0.50,
    teacher_forcing_end=0.00,
    detach_exposure_for_demand=False,
    grad_clip=1.0,
):
    opt = torch.optim.Adam(joint_model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(joint_epochs, 1))
    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(joint_epochs):
        if joint_epochs <= 1:
            tf_rate = teacher_forcing_end
        else:
            t = epoch / max(joint_epochs - 1, 1)
            tf_rate = teacher_forcing_start * (1.0 - t) + teacher_forcing_end * t

        joint_model.train()
        sums = {"loss": 0.0, "exp": 0.0, "dem": 0.0, "noexp": 0.0}
        nb = 0
        for b in tr_ld:
            b = _to_device_batch(b, device)
            out = joint_model(
                b["exp"], b["dem"], nZ_demand=nZ_demand,
                teacher_forcing_rate=tf_rate,
                detach_exposure_for_demand=detach_exposure_for_demand,
                return_aux=True,
            )
            exp_aux = out["exp_aux"]
            exp_loss = exp_ns["exposure_hurdle_loss"](
                exp_aux["log_hat"],
                b["exp"]["future_total_dph"],
                b["exp"]["future_buy_box_dph"],
                b["exp"]["future_instock_dph"],
                exp_aux["active_logit"],
                log_mag=exp_aux.get("log_mag", None),
            )
            dem_loss, _ = demand_loss_from_preds(
                dem_ns, b["dem"]["y"], out["demand_preds"], out["demand_z_reg"],
                beta_tail=beta_tail, lambda_under=lambda_under, lambda_z_reg=lambda_z_reg,
            )
            mu_mean = torch.stack([mu for mu, _ in out["demand_preds"]], dim=1).mean(dim=1)
            noexp_loss = no_exposure_no_demand_loss(out["pred_exp_log"], mu_mean, b["dem"]["y"])
            loss = lambda_exposure * exp_loss + lambda_demand * dem_loss + lambda_noexp * noexp_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(joint_model.parameters(), grad_clip)
            opt.step()

            sums["loss"] += float(loss.item())
            sums["exp"] += float(exp_loss.item())
            sums["dem"] += float(dem_loss.item())
            sums["noexp"] += float(noexp_loss.item())
            nb += 1

        sch.step()

        joint_model.eval()
        val = 0.0
        val_exp = 0.0
        val_dem = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = _to_device_batch(b, device)
                out = joint_model(
                    b["exp"], b["dem"], nZ_demand=max(4, min(nZ_demand, 8)),
                    teacher_forcing_rate=0.0,
                    detach_exposure_for_demand=False,
                    return_aux=True,
                )
                exp_aux = out["exp_aux"]
                exp_loss = exp_ns["exposure_hurdle_loss"](
                    exp_aux["log_hat"],
                    b["exp"]["future_total_dph"],
                    b["exp"]["future_buy_box_dph"],
                    b["exp"]["future_instock_dph"],
                    exp_aux["active_logit"],
                    log_mag=exp_aux.get("log_mag", None),
                )
                dem_loss, _ = demand_loss_from_preds(
                    dem_ns, b["dem"]["y"], out["demand_preds"], out["demand_z_reg"],
                    beta_tail=beta_tail, lambda_under=lambda_under, lambda_z_reg=lambda_z_reg,
                )
                # Main early stopping follows downstream demand but keeps exposure in the objective.
                vl = 0.25 * exp_loss + dem_loss
                val += float(vl.item())
                val_exp += float(exp_loss.item())
                val_dem += float(dem_loss.item())
        val /= max(1, len(va_ld)); val_exp /= max(1, len(va_ld)); val_dem /= max(1, len(va_ld))

        improved = val < best_val
        if improved:
            best_val = val
            best_sd = {k: v.detach().cpu().clone() for k, v in joint_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | train={sums['loss']/max(nb,1):.4f} "
            f"(exp={sums['exp']/max(nb,1):.4f}, dem={sums['dem']/max(nb,1):.4f}, noexp={sums['noexp']/max(nb,1):.4f}) | "
            f"val={val:.4f} (exp={val_exp:.4f}, dem={val_dem:.4f}) | tf={tf_rate:.2f}"
            + (" *" if improved else "")
        )
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}; best_val={best_val:.4f}")
            break

    if best_sd is not None:
        joint_model.load_state_dict(best_sd)
    return joint_model


def generate_joint_forecasts(joint_model, va_ld, device, M=80, exposure_hat_csv_path="exposure_hat_for_demand.csv"):
    rows = []
    hat_rows = []
    joint_model.eval()
    with torch.no_grad():
        for b0 in va_ld:
            b = _to_device_batch(b0, device)
            # exposure prediction
            exp_aux = joint_model.exposure_model(b["exp"]["x"], b["exp"]["future_context"], return_aux=True)
            pred_log = exp_aux["log_hat"]
            pred_level = torch.expm1(pred_log).clamp(min=0.0)

            fc_dem = b["dem"]["future_context"].clone()
            fc_dem[:, :, -3:] = pred_log
            p50, p70, stock_log_hat = joint_model.demand_model.predict(
                b["dem"]["x"], fc_dem, M=M, return_stock=True
            )
            hist_mean = (b["dem"]["x"][:, :, 0].exp() - 1).mean(dim=1, keepdim=True).clamp(min=0)
            hm50 = hist_mean.expand_as(b["dem"]["y"])
            hm70 = hm50 * 1.25
            B, H = b["dem"]["y"].shape
            for i in range(B):
                asin = b0["dem"]["asin"][i]
                for h in range(H):
                    wk = pd.to_datetime(b0["dem"]["target_week"][h][i])
                    y = b["dem"]["y"][i, h].item()
                    price = b["dem"].get("our_price", torch.zeros_like(b["dem"]["y"]))[i, h].item()
                    pkg = b["dem"].get("pkg_volume", torch.zeros_like(b["dem"]["y"]))[i, h].item()
                    rows.append({
                        "asin": asin,
                        "order_week": wk,
                        "fcst_week_index": h + 1,
                        "fbi_demand": y,
                        "our_price": price,
                        "true_amt": y * price,
                        "pkg_volume": pkg,
                        "true_size": y * pkg,
                        "true_future_total_dph": b["exp"]["future_total_dph"][i, h].item(),
                        "true_future_buy_box_dph": b["exp"]["future_buy_box_dph"][i, h].item(),
                        "true_future_instock": b["exp"]["future_instock_dph"][i, h].item(),
                        "pred_total_dph_hat": pred_level[i, h, 0].item(),
                        "pred_buy_box_dph_hat": pred_level[i, h, 1].item(),
                        "pred_instock_dph_hat": pred_level[i, h, 2].item(),
                        "scot_oos": b["dem"].get("oos", torch.zeros_like(b["dem"]["y"]))[i, h].item(),
                        "oos": b["dem"].get("oos", torch.zeros_like(b["dem"]["y"]))[i, h].item(),
                        "p50_amxl": p50[i, h].item(),
                        "p70_amxl": p70[i, h].item(),
                        "p50_scot": hm50[i, h].item(),
                        "p70_scot": hm70[i, h].item(),
                    })
                    hat_rows.append({
                        "asin": asin,
                        "order_week": wk,
                        "pred_total_dph": pred_level[i, h, 0].item(),
                        "pred_buy_box_dph": pred_level[i, h, 1].item(),
                        "pred_instock_dph": pred_level[i, h, 2].item(),
                    })
    forecast_df = pd.DataFrame(rows)
    hat_df = pd.DataFrame(hat_rows)
    if len(hat_df) > 0:
        hat_df = (hat_df.groupby(["asin", "order_week"], as_index=False)
                  .agg(pred_total_dph=("pred_total_dph", "mean"),
                       pred_buy_box_dph=("pred_buy_box_dph", "mean"),
                       pred_instock_dph=("pred_instock_dph", "mean")))
        hat_df.to_csv(exposure_hat_csv_path, index=False)
        print(f"Saved joint exposure hats: {exposure_hat_csv_path} | rows={len(hat_df):,}")
    return forecast_df, hat_df


# -------------------------------
# Main runner
# -------------------------------

def run_joint_exposure_demand_end2end_v2(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    exposure_history=13,
    demand_history=52,
    horizon=20,
    batch_size=64,
    patience=8,
    pretrain_exposure_epochs=8,
    joint_epochs=40,
    exposure_script=None,
    demand_script=None,
    exposure_hat_csv_path="exposure_hat_for_demand.csv",
    seed=42,
    device=None,
    lr_joint=5e-4,
    lambda_exposure=1.0,
    lambda_demand=1.0,
    lambda_noexp=0.03,
    teacher_forcing_start=0.50,
    teacher_forcing_end=0.00,
    detach_exposure_for_demand=False,
    M_eval=80,
    use_encoder_self_attn=True,
    d_model_exposure=64,
    d_model_demand=32,
    z_dim_exposure=8,
    z_dim_demand=16,
    **kwargs,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"End-to-end demand gradient into exposure: {not detach_exposure_for_demand}")

    exposure_script = exposure_script or "tcn_exposure_v2_enn_regime_singlehead_nogate_peakloss_gpu_GRAPH_HEAD_FUSION_PEAKCROSS_ROUTER_v27_8c_DECODER_ZERO_ATTN_ACTIVEPROTECTED_PREDICTFIX_LOCALCSV.py"
    demand_script = demand_script or "demand_external_exposure3_clean_3modes_GRAPH_KNOWNPROMO_ZDECODER_ACTIVELOSS_HDIAG_EXPGATE_READHAT_NOQ_v12d_LOCALCSV.py"
    exp_ns = runpy.run_path(_find_script(exposure_script))
    dem_ns = runpy.run_path(_find_script(demand_script))

    # Same sample policy as current demand scripts: sample then intersect with SCOT if available.
    if scot_df is not None and "prepare_data_from_sample_scot_intersection" in dem_ns:
        data_use = dem_ns["prepare_data_from_sample_scot_intersection"](
            data_raw1, scot_df, n_asins=n_asins, seed=seed
        )
    else:
        # fallback: take first/random n_asins dict entries
        if isinstance(data_raw1, dict):
            keys = list(data_raw1.keys())[:n_asins]
            data_use = {k: data_raw1[k] for k in keys}
        else:
            data_use = data_raw1

    print("\nLoading exposure data...")
    exp_data, exp_context_dim, exp_context_cols = exp_ns["load_exposure_data"](data_use)
    print("\nLoading demand data...")
    dem_data, dem_context_dim, dem_context_cols = dem_ns["load_real_data"](data_use)

    exp_tr = exp_ns["ExposureDataset"](exp_data, history=exposure_history, horizon=horizon, mode="train", val_weeks=horizon)
    exp_va = exp_ns["ExposureDataset"](exp_data, history=exposure_history, horizon=horizon, mode="val", val_weeks=horizon)
    dem_tr = dem_ns["DemandDataset"](dem_data, history=demand_history, horizon=horizon, mode="train", val_weeks=horizon)
    dem_va = dem_ns["DemandDataset"](dem_data, history=demand_history, horizon=horizon, mode="val", val_weeks=horizon)

    pair_tr = PairedExposureDemandDataset(exp_tr, dem_tr)
    pair_va = PairedExposureDemandDataset(exp_va, dem_va)
    tr_ld = DataLoader(pair_tr, batch_size=batch_size, shuffle=True, collate_fn=paired_collate, drop_last=False)
    va_ld = DataLoader(pair_va, batch_size=batch_size, shuffle=False, collate_fn=paired_collate, drop_last=False)

    # Build models.
    exp_input_dim = next(iter(exp_data.values()))["features"].shape[1]
    dem_input_dim = next(iter(dem_data.values()))["features"].shape[1]
    exposure_model = exp_ns["ExposureForecastModelV2"](
        input_dim=exp_input_dim,
        context_dim=exp_context_dim,
        d_model=d_model_exposure,
        horizon=horizon,
        context_cols=exp_context_cols,
        use_encoder_self_attn=use_encoder_self_attn,
        use_enn=True,
        z_dim=z_dim_exposure,
    )
    demand_model = dem_ns["TCN_ENN"](
        input_dim=dem_input_dim,
        context_dim=dem_context_dim,
        d_model=d_model_demand,
        d_z=z_dim_demand,
        horizon=horizon,
        use_exposure_demand_gate=True,
    )
    joint_model = JointExposureDemandModel(exposure_model, demand_model).to(device)

    # Optional exposure pretrain inside the joint paired loader.
    if pretrain_exposure_epochs and pretrain_exposure_epochs > 0:
        print("\n" + "=" * 100)
        print(f"PRETRAIN EXPOSURE BRANCH ONLY: {pretrain_exposure_epochs} epochs")
        print("=" * 100)
        opt = torch.optim.Adam(joint_model.exposure_model.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(pretrain_exposure_epochs):
            joint_model.train()
            s = 0.0; nb = 0
            for b in tr_ld:
                b = _to_device_batch(b, device)
                exp_aux = joint_model.exposure_model(b["exp"]["x"], b["exp"]["future_context"], return_aux=True)
                loss = exp_ns["exposure_hurdle_loss"](
                    exp_aux["log_hat"],
                    b["exp"]["future_total_dph"],
                    b["exp"]["future_buy_box_dph"],
                    b["exp"]["future_instock_dph"],
                    exp_aux["active_logit"],
                    log_mag=exp_aux.get("log_mag", None),
                )
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(joint_model.exposure_model.parameters(), 1.0); opt.step()
                s += float(loss.item()); nb += 1
            print(f"Exposure pretrain epoch {ep+1:3d} | loss={s/max(nb,1):.4f}")

    print("\n" + "=" * 100)
    print("JOINT END-TO-END TRAINING")
    print("=" * 100)
    joint_model = train_joint_end2end(
        joint_model, exp_ns, dem_ns, tr_ld, va_ld, device=device,
        joint_epochs=joint_epochs,
        lr=lr_joint,
        patience=patience,
        lambda_exposure=lambda_exposure,
        lambda_demand=lambda_demand,
        lambda_noexp=lambda_noexp,
        teacher_forcing_start=teacher_forcing_start,
        teacher_forcing_end=teacher_forcing_end,
        detach_exposure_for_demand=detach_exposure_for_demand,
    )

    print("\nGenerating joint validation forecasts...")
    forecast_df, exposure_hat_df = generate_joint_forecasts(
        joint_model, va_ld, device=device, M=M_eval, exposure_hat_csv_path=exposure_hat_csv_path
    )
    forecast_df = _merge_scot_if_available(forecast_df, scot_df)
    wape_summary = _simple_wape_summary(forecast_df, remove_oos_dp=True)

    summary = {
        "model": "joint_exposure_demand_v2_end2end",
        "end_to_end": bool(not detach_exposure_for_demand),
        "n_asins": n_asins,
        "train_pairs": len(pair_tr),
        "val_pairs": len(pair_va),
        "exposure_history": exposure_history,
        "demand_history": demand_history,
        "horizon": horizon,
        "pretrain_exposure_epochs": pretrain_exposure_epochs,
        "joint_epochs_requested": joint_epochs,
        "teacher_forcing_start": teacher_forcing_start,
        "teacher_forcing_end": teacher_forcing_end,
        "lambda_exposure": lambda_exposure,
        "lambda_demand": lambda_demand,
        "lambda_noexp": lambda_noexp,
        "exposure_hat_csv_path": exposure_hat_csv_path,
    }
    summary.update(wape_summary)
    _print_summary(summary)

    return {
        "joint_model": joint_model,
        "forecast_df": forecast_df,
        "exposure_hat_for_demand": exposure_hat_df,
        "wape_summary": wape_summary,
        "joint_summary": summary,
    }


# Backward-friendly alias
run_joint_exposure_demand_v2 = run_joint_exposure_demand_end2end_v2
