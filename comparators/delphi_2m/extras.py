# Delphi-2M-inspired — Supported-horizon integrated summaries (12–84 and 12–96 months) and optional validation-recalibrated sensitivity
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 37 ---
def export_supported_window_integrated_summaries(
    run_dir: Path,
    end_horizons_months: Sequence[int] = (84, 96),
) -> Dict[int, pd.DataFrame]:
    dense_path = Path(run_dir) / "test_dense_metrics_by_condition_and_horizon.csv"
    if not dense_path.is_file():
        print("Skipping supported-horizon summary (dense metrics file not found).")
        return {}
    dense = pd.read_csv(dense_path)
    dense = dense[dense["risk_type"].astype(str).eq("death_adjusted")].copy()
    trapezoid = getattr(np, "trapezoid", np.trapz)
    outputs: Dict[int, pd.DataFrame] = {}
    macro_rows: List[Dict[str, Any]] = []

    for max_h in end_horizons_months:
        sub = dense[dense["horizon_months"] <= int(max_h)].copy()
        cond_rows: List[Dict[str, Any]] = []
        for (cid, cname), grp in sub.sort_values("horizon_months").groupby(
            ["condition_id", "condition_name"], sort=True
        ):
            x = grp["horizon_months"].to_numpy(float)
            span = float(x[-1] - x[0])
            if span <= 0:
                continue
            ibs = float(trapezoid(grp["brier_ipcw_ht"].to_numpy(float), x) / span)
            ibs_null = float(trapezoid(grp["brier_null_ipcw_ht"].to_numpy(float), x) / span)
            inbll = float(trapezoid(grp["nbll_ipcw_ht"].to_numpy(float), x) / span)
            auc_valid = grp["auc_ipcw_cr"].notna()
            iauc = (
                float(trapezoid(grp.loc[auc_valid, "auc_ipcw_cr"].to_numpy(float), x[auc_valid]) / (x[auc_valid][-1] - x[auc_valid][0]))
                if int(auc_valid.sum()) >= 2
                else np.nan
            )
            ical = float(trapezoid(grp["absolute_calibration_error"].to_numpy(float), x) / span)
            cond_rows.append({
                "condition_id": int(cid),
                "condition_name": str(cname),
                "integration_start_months": int(x[0]),
                "integration_end_months": int(x[-1]),
                "n_grid_points": int(len(x)),
                "iauc_ipcw_cr": iauc,
                "ibs_ipcw_ht": ibs,
                "ibs_null_ipcw_ht": ibs_null,
                "scaled_ibs_ipcw_ht": (1.0 - ibs / ibs_null) if ibs_null > 0 else np.nan,
                "inbll_ipcw_ht": inbll,
                "integrated_abs_calibration_error": ical,
            })
        cond_df = pd.DataFrame(cond_rows)
        out_csv = Path(run_dir) / f"test_integrated_metrics_12_to_{int(max_h)}m_delphi.csv"
        cond_df.to_csv(out_csv, index=False)
        outputs[int(max_h)] = cond_df
        macro_rows.append({
            "window_months": f"12-{int(max_h)}",
            "n_conditions": int(len(cond_df)),
            "macro_iauc_ipcw_cr": float(cond_df["iauc_ipcw_cr"].mean()),
            "macro_ibs_ipcw_ht": float(cond_df["ibs_ipcw_ht"].mean()),
            "macro_ibs_null_ipcw_ht": float(cond_df["ibs_null_ipcw_ht"].mean()),
            "macro_scaled_ibs_ipcw_ht": float(cond_df["scaled_ibs_ipcw_ht"].mean()),
            "macro_inbll_ipcw_ht": float(cond_df["inbll_ipcw_ht"].mean()),
            "macro_integrated_abs_calibration_error": float(
                cond_df["integrated_abs_calibration_error"].mean()
            ),
        })
    if macro_rows:
        macro_df = pd.DataFrame(macro_rows)
        macro_df.to_csv(
            Path(run_dir) / "test_integrated_macro_supported_windows_delphi.csv",
            index=False,
        )
        print("Supported-horizon macro integrated metrics (Delphi):")
        print(macro_df.to_string(index=False))
    return outputs


_ = export_supported_window_integrated_summaries(RUN_DIR, end_horizons_months=(84, 96))

# --- Notebook cell 39 ---
RUN_RECALIBRATED_SENSITIVITY = _env_bool(
    "EHR_DELPHI_RUN_RECALIBRATED_SENSITIVITY", "YES"
)


@torch.no_grad()
def run_validation_recalibrated_sensitivity(
    checkpoint_path: Path,
) -> Optional[pd.DataFrame]:
    """Fit validation-only rate multipliers and score locked test into a sidecar folder."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        print("Skipping recalibrated sensitivity (checkpoint not found).")
        return None
    sens_dir = RUN_DIR / "validation_recalibrated_sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    model = load_frozen_model(checkpoint_path)
    model.eval()
    starts = np.arange(N_BINS, dtype=np.float64) * float(BIN_WIDTH_MONTHS)
    first_bin = int(CFG.gap_months // BIN_WIDTH_MONTHS)

    condition_expected = np.zeros(K, dtype=np.float64)
    condition_events = np.zeros(K, dtype=np.float64)
    death_expected = 0.0
    death_events = 0.0

    for start_row in range(0, len(VAL_SAMPLES), CFG.batch_size):
        stop_row = min(start_row + CFG.batch_size, len(VAL_SAMPLES))
        frame = VAL_SAMPLES.iloc[start_row:stop_row]
        rows = [frame.iloc[i] for i in range(len(frame))]
        cond_r, death_r = predict_rate_grid_for_rows(model, rows, apply_rate_scaling=False)
        cond_np = cond_r.numpy().astype(np.float64)
        death_np = death_r.numpy().astype(np.float64)

        val_out = make_condition_outcomes(frame)
        elig = val_out["eligible"]
        stop_from_idx = np.clip(
            CFG.gap_months + val_out["exit_time_untruncated"], 0.0, float(MAX_HORIZON_MONTHS)
        )
        exp_bins = np.clip(
            stop_from_idx[..., None] - starts[None, None, :], 0.0, float(BIN_WIDTH_MONTHS)
        ) * elig[..., None]
        condition_expected += np.sum(cond_np[..., first_bin:] * exp_bins[..., first_bin:], axis=(0, 2))
        condition_events += np.sum(
            elig & (val_out["status"] == 1) & (stop_from_idx <= float(MAX_HORIZON_MONTHS) + TIME_EPS_MONTHS),
            axis=0,
        )

        val_d = make_death_outcomes(frame)
        d_elig = val_d["eligible"]
        d_stop = np.clip(
            CFG.gap_months + val_d["exit_time_untruncated"], 0.0, float(MAX_HORIZON_MONTHS)
        )
        d_exp = np.clip(
            d_stop[:, None] - starts[None, :], 0.0, float(BIN_WIDTH_MONTHS)
        ) * d_elig[:, None]
        death_expected += float(np.sum(death_np[:, first_bin:] * d_exp[:, first_bin:]))
        death_events += float(np.sum(
            d_elig & (val_d["status"] == 1) & (d_stop <= float(MAX_HORIZON_MONTHS) + TIME_EPS_MONTHS)
        ))

    prior = float(CFG.rate_scaling_prior_events)
    lower = math.exp(-float(CFG.rate_scaling_log_limit))
    upper = math.exp(float(CFG.rate_scaling_log_limit))
    cond_scale = np.clip((condition_events + prior) / (condition_expected + prior), lower, upper)
    death_scale = float(np.clip((death_events + prior) / (death_expected + prior), lower, upper))
    atomic_json_dump(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_split": "validation_only_post_entry",
            "prior_events": prior,
            "condition_scale_median": float(np.median(cond_scale)),
            "condition_scale_min": float(np.min(cond_scale)),
            "condition_scale_max": float(np.max(cond_scale)),
            "death_scale": death_scale,
            "condition_scale": cond_scale.tolist(),
        },
        sens_dir / "validation_post_entry_rate_scales.json",
    )

    test_df = load_canonical_test_samples()
    n_test = len(test_df)
    grid_n = len(DENSE_HORIZONS_MONTHS)
    adj_recal = np.empty((n_test, K, grid_n), dtype=np.float32)
    unadj_recal = np.empty((n_test, K, grid_n), dtype=np.float32)
    death_recal = np.empty((n_test, grid_n), dtype=np.float32)
    cond_scale_t = torch.as_tensor(cond_scale, dtype=torch.float32).view(1, K, 1)

    for start in range(0, n_test, CFG.batch_size):
        stop = min(start + CFG.batch_size, n_test)
        rows = [test_df.iloc[i] for i in range(start, stop)]
        cond_r, death_r = predict_rate_grid_for_rows(model, rows, apply_rate_scaling=False)
        cond_r = cond_r * cond_scale_t
        death_r = death_r * death_scale
        for g_idx, h in enumerate(DENSE_HORIZONS_MONTHS):
            ra, ru, rd = conditional_window_risks_from_rates(cond_r, death_r, int(h))
            adj_recal[start:stop, :, g_idx] = ra.numpy()
            unadj_recal[start:stop, :, g_idx] = ru.numpy()
            death_recal[start:stop, g_idx] = rd.numpy()

    recal_preds = {"adjusted": adj_recal, "unadjusted": unadj_recal, "death": death_recal}
    recal_res = evaluate_all_metrics(test_df, recal_preds)
    recal_compat = build_main_compatibility_tables(recal_res, test_df, recal_preds)
    recal_compat["summary"].to_csv(sens_dir / "test_summary_publication_recalibrated.csv", index=False)
    recal_res["dense"].to_csv(sens_dir / "test_dense_metrics_by_condition_and_horizon.csv", index=False)
    recal_res["integrated"].query("risk_type == 'death_adjusted'").to_csv(
        sens_dir / "test_integrated_metrics_by_condition.csv", index=False
    )
    print("\nValidation-recalibrated Delphi sensitivity summary (death-adjusted):")
    summary_da = recal_compat["summary"][recal_compat["summary"]["risk_type"].eq("death_adjusted")]
    print(summary_da.to_string(index=False))
    _ = export_supported_window_integrated_summaries(sens_dir, end_horizons_months=(84, 96))
    return summary_da


if RUN_RECALIBRATED_SENSITIVITY:
    _recal_ckpt = Path(globals().get("SELECTED_CHECKPOINT", resolved_checkpoint_path()))
    _ = run_validation_recalibrated_sensitivity(_recal_ckpt)
else:
    print("Skipped validation-recalibrated sensitivity cell.")
