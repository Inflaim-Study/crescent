# Delphi-2M-inspired — Defines autoregressive training (50,000 updates) and checkpoint selection
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 19 ---
def cosine_learning_rate(iteration: int) -> float:
    if iteration < CFG.warmup_iterations:
        return CFG.learning_rate * float(iteration + 1) / max(
            1, CFG.warmup_iterations
        )
    if iteration >= CFG.max_iterations:
        return CFG.min_learning_rate
    ratio = (
        iteration - CFG.warmup_iterations
    ) / max(1, CFG.max_iterations - CFG.warmup_iterations)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return CFG.min_learning_rate + coefficient * (
        CFG.learning_rate - CFG.min_learning_rate
    )


def configure_delphi_optimizer(
    model: nn.Module,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if parameter_name.endswith("bias"):
                no_decay.append(parameter)
            elif isinstance(module, nn.Linear) and parameter_name == "weight":
                decay.append(parameter)
            else:
                no_decay.append(parameter)
    # Tied parameters can appear through two modules; retain one identity only.
    def unique(parameters: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        result, seen = [], set()
        for parameter in parameters:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
        return result
    no_decay = unique(no_decay)
    no_decay_ids = {id(parameter) for parameter in no_decay}
    decay = [
        parameter
        for parameter in unique(decay)
        if id(parameter) not in no_decay_ids
    ]
    covered = {id(parameter) for parameter in decay + no_decay}
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if covered != expected:
        raise RuntimeError("Optimizer parameter grouping is incomplete.")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": CFG.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=CFG.learning_rate,
        betas=(CFG.beta1, CFG.beta2),
    )


@torch.no_grad()
def evaluate_ar_nll(
    model: nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    model.eval()
    ar_keys = (
        "nll_numerator", "token_nll_numerator", "time_nll_numerator",
        "censor_nll_numerator", "token_count", "censor_interval_count",
        "contribution_count", "no_event_target_count",
        "disease_death_target_count",
    )
    full_totals = {key: 0.0 for key in ar_keys}
    selection_totals = {key: 0.0 for key in ar_keys}
    for batch_index, raw_batch in enumerate(loader):
        if CFG.validation_batches > 0 and batch_index >= CFG.validation_batches:
            break
        batch = move_batch(raw_batch, DEVICE)
        with torch.no_grad(), autocast_context(DEVICE, PRECISION):
            outputs = model(batch, training_targets=True)
            full_terms = delphi_likelihood_terms(
                outputs,
                batch,
                exclude_no_event_targets=False,
                include_censor_survival=False,
            )
            selection_terms = delphi_likelihood_terms(
                outputs,
                batch,
                exclude_no_event_targets=CFG.selection_exclude_no_event,
                include_censor_survival=False,
            )
        for key in ar_keys:
            full_totals[key] += float(full_terms[key].detach().cpu())
            selection_totals[key] += float(
                selection_terms[key].detach().cpu()
            )
    if full_totals["contribution_count"] <= 0:
        raise RuntimeError("Validation loader produced no Delphi contributions.")
    if selection_totals["contribution_count"] <= 0:
        raise RuntimeError(
            "Validation loader produced no original Delphi selection events."
        )
    selection_ar_nll = (
        selection_totals["nll_numerator"]
        / selection_totals["contribution_count"]
    )
    return {
        "selection_joint_nll": selection_ar_nll,
        "selection_original_delphi_nll": selection_ar_nll,
        "selection_token_nll": (
            selection_totals["token_nll_numerator"]
            / max(selection_totals["token_count"], 1.0)
        ),
        "selection_time_nll": (
            selection_totals["time_nll_numerator"]
            / max(selection_totals["token_count"], 1.0)
        ),
        "full_joint_nll": (
            full_totals["nll_numerator"]
            / full_totals["contribution_count"]
        ),
        "full_token_nll": (
            full_totals["token_nll_numerator"]
            / max(full_totals["token_count"], 1.0)
        ),
        "full_time_nll": (
            full_totals["time_nll_numerator"]
            / max(full_totals["token_count"], 1.0)
        ),
        "full_censor_nll": 0.0,
        **{f"full_{key}": value for key, value in full_totals.items()},
        **{f"selection_{key}": value for key, value in selection_totals.items()},
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    iteration: int,
    best_validation_original_delphi_nll: float,
    validations_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    architecture: Mapping[str, Any],
    raw_model_state: Optional[Mapping[str, Any]] = None,
    weights_type: str = "raw",
) -> Dict[str, Any]:
    return {
        "schema_version": CFG.schema_version,
        "iteration": int(iteration),
        "weights_type": str(weights_type),
        "ema_decay": float(CFG.ema_decay),
        "model_state": model.state_dict(),
        "raw_model_state": None if raw_model_state is None else dict(raw_model_state),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "best_validation_original_delphi_nll": float(
            best_validation_original_delphi_nll
        ),
        "validations_without_improvement": int(
            validations_without_improvement
        ),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "config": asdict(CFG),
        "scientific_config_sha256": canonical_json_hash(
            MODEL_SCIENTIFIC_CONFIG
        ),
        "architecture": dict(architecture),
        "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_HASH,
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "canonical_train_samples_sha256": sha256_file(
            CANONICAL_PATHS["train"]
        ),
        "canonical_val_samples_sha256": sha256_file(
            CANONICAL_PATHS["val"]
        ),
    }


def train_model() -> Path:
    set_seed(CFG.training_seed)
    best_path = RUN_DIR / "best_checkpoint.pt"
    last_path = RUN_DIR / "last_checkpoint.pt"
    history_path = RUN_DIR / "training_history.csv"
    if best_path.exists() and not CFG.resume_checkpoint:
        raise FileExistsError(
            "Training artifacts already exist. Use a new EHR_DELPHI_OUTPUT "
            "or set EHR_DELPHI_RESUME."
        )

    from contextlib import contextmanager

    model = DelphiCanonicalModel().to(DEVICE)
    optimizer = configure_delphi_optimizer(model)
    scaler = (
        torch.cuda.amp.GradScaler()
        if DEVICE.type == "cuda" and PRECISION == "fp16"
        else None
    )
    use_ema = float(CFG.ema_decay) > 0.0
    ema_params = [p.detach().clone() for p in model.parameters()] if use_ema else []

    @torch.no_grad()
    def ema_update(step_iteration: int) -> None:
        if not use_ema:
            return
        decay = min(float(CFG.ema_decay), (1.0 + float(step_iteration)) / (10.0 + float(step_iteration)))
        for e, p in zip(ema_params, model.parameters()):
            e.lerp_(p.detach(), 1.0 - decay)

    @contextmanager
    def ema_weights_in_model():
        if not use_ema:
            yield
            return
        backup = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, e in zip(model.parameters(), ema_params):
                p.copy_(e)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(model.parameters(), backup):
                    p.copy_(b)

    train_dataset = DelphiARDataset(TRAIN_SAMPLES, "train")
    val_loader = make_ar_loader(VAL_SAMPLES, "val", shuffle=False)
    start_iteration = 1
    best_validation_original_delphi_nll = math.inf
    validations_without_improvement = 0
    history: List[Dict[str, Any]] = []

    if CFG.resume_checkpoint:
        checkpoint = torch_load_compat(
            Path(CFG.resume_checkpoint).expanduser().resolve(), DEVICE
        )
        if checkpoint.get("preprocessing_contract_sha256") != PREPROCESSING_CONTRACT_HASH:
            raise RuntimeError("Resume checkpoint has a different preprocessing contract.")
        if checkpoint.get("scientific_config_sha256") != canonical_json_hash(
            MODEL_SCIENTIFIC_CONFIG
        ):
            raise RuntimeError("Resume checkpoint scientific configuration mismatch.")
        if use_ema and checkpoint.get("raw_model_state") is not None:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            ema_params = [p.detach().clone() for p in model.parameters()]
            model.load_state_dict(checkpoint["raw_model_state"], strict=True)
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            if use_ema:
                ema_params = [p.detach().clone() for p in model.parameters()]
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_iteration = int(checkpoint["iteration"]) + 1
        best_validation_original_delphi_nll = float(
            checkpoint["best_validation_original_delphi_nll"]
        )
        validations_without_improvement = int(
            checkpoint["validations_without_improvement"]
        )
        history = list(checkpoint.get("history", []))
        if "rng_state" not in checkpoint:
            raise RuntimeError("Resume checkpoint lacks the deterministic RNG state.")
        restore_rng_state(checkpoint["rng_state"])

    architecture = model_architecture_manifest(model)
    if not architecture["embedding_output_weight_tying"]:
        raise RuntimeError("Published Delphi weight tying is not active.")
    atomic_json_dump(architecture, RUN_DIR / "model_architecture.json")
    atomic_json_dump({
        "schema_version": CFG.schema_version,
        "started_unix_time": time.time(),
        "device": str(DEVICE),
        "precision": PRECISION,
        "train_participants": len(TRAIN_SAMPLES),
        "validation_participants": len(VAL_SAMPLES),
        "validation_discrimination_participants": len(VAL_SAMPLES),
        "validation_discrimination_full_cohort": True,
        "validation_discrimination_schedule": (
            f"every {CFG.validation_risk_report_interval:,} updates plus selected checkpoint"
            if CFG.log_validation_metrics
            else "selected checkpoint once after checkpoint selection"
        ),
        "test_loaded": False,
        "checkpoint_selection": (
            "minimum original Delphi validation next-event plus waiting-time loss"
        ),
    }, RUN_DIR / "training_started.json")

    ar_keys = (
        "nll_numerator", "token_nll_numerator", "time_nll_numerator",
        "censor_nll_numerator", "token_count", "censor_interval_count",
        "contribution_count", "no_event_target_count",
        "disease_death_target_count",
    )
    running_ar = {key: 0.0 for key in ar_keys}
    discrimination_history_path = (
        RUN_DIR / "validation_discrimination_history.csv"
    )
    validation_discrimination_history: List[Dict[str, Any]] = []
    if CFG.resume_checkpoint and discrimination_history_path.is_file():
        validation_discrimination_history = pd.read_csv(
            discrimination_history_path
        ).to_dict(orient="records")
    try:
        if CFG.run_preflight_batch:
            preflight = move_batch(
                deterministic_training_batch(train_dataset, 1), DEVICE
            )
            model.eval()
            with torch.no_grad(), autocast_context(DEVICE, PRECISION):
                preflight_ar = delphi_likelihood_terms(
                    model(preflight, training_targets=True),
                    preflight,
                    exclude_no_event_targets=False,
                    include_censor_survival=False,
                )
                preflight_objective = preflight_ar["objective"]
            if not torch.isfinite(preflight_objective):
                raise FloatingPointError("Preflight batch produced non-finite loss.")
            atomic_json_dump({
                "passed": True,
                "ar_joint_nll": float(preflight_ar["objective"].detach().cpu()),
                "training_objective": "original_delphi_event_plus_time_nll",
                "combined_objective": float(preflight_objective.detach().cpu()),
                "landmark_auxiliary_weight": 0.0,
                "batch_size": len(preflight["eid"]),
            }, RUN_DIR / "training_preflight.json")

        for iteration in range(start_iteration, CFG.max_iterations + 1):
            started = time.time()
            batch = move_batch(
                deterministic_training_batch(train_dataset, iteration), DEVICE
            )
            model.train()
            learning_rate = cosine_learning_rate(iteration - 1)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(DEVICE, PRECISION):
                ar_terms = delphi_likelihood_terms(
                    model(batch, training_targets=True),
                    batch,
                    exclude_no_event_targets=False,
                    include_censor_survival=False,
                )
                loss = ar_terms["objective"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Delphi canonical-index loss at iteration {iteration}."
                )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.max_gradient_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG.max_gradient_norm)
                optimizer.step()
            ema_update(iteration)
            for key in ar_keys:
                running_ar[key] += float(ar_terms[key].detach().cpu())

            should_validate = (
                iteration % CFG.validation_interval == 0
                or iteration == CFG.max_iterations
            )
            if iteration % 100 == 0 and not should_validate:
                print(
                    f"Training iteration {iteration:06d}/{CFG.max_iterations:06d} "
                    f"| current AR NLL={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if not should_validate:
                continue

            print(
                f"Training iteration {iteration:06d} complete; "
                "running checkpoint-selection AR validation.",
                flush=True,
            )
            with ema_weights_in_model():
                validation = evaluate_ar_nll(model, val_loader)
                discrimination_frame = pd.DataFrame()
                should_report_risk = (
                    CFG.log_validation_metrics
                    and (
                        iteration % CFG.validation_risk_report_interval == 0
                        or iteration == CFG.max_iterations
                    )
                )
                if should_report_risk:
                    print(
                        "Running the full-cohort validation risk grid "
                        f"at iteration {iteration:06d}.",
                        flush=True,
                    )
                    discrimination_metrics, discrimination_frame = (
                        validation_risk_reporting_metrics(
                            model,
                            VAL_SAMPLES,
                            iteration=int(iteration),
                            is_selected_checkpoint=False,
                        )
                    )
                    validation.update(discrimination_metrics)
            if use_ema:
                validation["val_original_delphi_nll_raw_weights"] = float(
                    evaluate_ar_nll(model, val_loader)["selection_original_delphi_nll"]
                )
            selection_score = validation["selection_original_delphi_nll"]
            validation["selection_score"] = float(selection_score)
            train_ar_nll = running_ar["nll_numerator"] / max(
                running_ar["contribution_count"], 1.0
            )
            improved_beyond_delta = (
                np.isfinite(selection_score)
                and selection_score
                < best_validation_original_delphi_nll - CFG.min_delta
            )
            improved = (
                np.isfinite(selection_score)
                and selection_score < best_validation_original_delphi_nll
            )
            if improved_beyond_delta:
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if improved:
                best_validation_original_delphi_nll = float(selection_score)
            row = {
                "iteration": int(iteration),
                "learning_rate": learning_rate,
                "train_joint_nll": train_ar_nll,
                "train_token_nll": (
                    running_ar["token_nll_numerator"]
                    / max(running_ar["token_count"], 1.0)
                ),
                "train_time_nll": (
                    running_ar["time_nll_numerator"]
                    / max(running_ar["token_count"], 1.0)
                ),
                "train_no_event_target_count": running_ar["no_event_target_count"],
                "train_disease_death_target_count": running_ar[
                    "disease_death_target_count"
                ],
                **validation,
                "best_validation_original_delphi_nll": (
                    best_validation_original_delphi_nll
                ),
                "validations_without_improvement": validations_without_improvement,
                "is_best": bool(improved),
                "seconds_since_last_iteration": time.time() - started,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)
            if not discrimination_frame.empty:
                discrimination_frame.insert(0, "iteration", int(iteration))
                validation_discrimination_history.extend(
                    discrimination_frame.to_dict(orient="records")
                )
                pd.DataFrame(validation_discrimination_history).to_csv(
                    discrimination_history_path, index=False
                )
            raw_state_for_resume = (
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if use_ema
                else None
            )
            with ema_weights_in_model():
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    scaler,
                    iteration,
                    best_validation_original_delphi_nll,
                    validations_without_improvement,
                    history,
                    architecture,
                    raw_model_state=raw_state_for_resume,
                    weights_type="ema" if use_ema else "raw",
                )
                atomic_torch_save(payload, last_path)
                if improved:
                    atomic_torch_save(payload, best_path)
            atomic_json_dump(row, RUN_DIR / "latest_validation.json")
            validation_summary = (
                f"Iteration {iteration:06d} | train AR NLL={train_ar_nll:.6f} "
                f"| val original Delphi NLL={selection_score:.6f}"
            )
            if not discrimination_frame.empty:
                validation_summary += (
                    " | val mean IPCW macro AUROC="
                    f"{validation['validation_macro_auc_ipcw_cr']:.4f}"
                    " | val mean IPCW macro AUPRC="
                    f"{validation['validation_macro_auprc_ipcw_cr']:.4f}"
                )
            validation_summary += f" | {'BEST' if improved else '--'}"
            print(validation_summary, flush=True)
            if not discrimination_frame.empty:
                print(
                    discrimination_frame[
                        [
                            "horizon_months",
                            "macro_auc_ipcw_cr",
                            "macro_auprc_ipcw_cr",
                            "conditions_with_auc",
                            "conditions_with_auprc",
                            "n_validation_participants",
                            "n_eligible_pairs",
                            "n_known_pairs",
                            "n_positive_pairs",
                            "n_competing_pairs",
                        ]
                    ].to_string(index=False),
                    flush=True,
                )
            running_ar = {key: 0.0 for key in ar_keys}
            if (
                CFG.patience_validations > 0
                and validations_without_improvement >= CFG.patience_validations
            ):
                print("Validation early stopping triggered.", flush=True)
                break

        if not best_path.is_file():
            raise RuntimeError("Training ended without a validation-selected checkpoint.")
        selected_checkpoint = torch_load_compat(best_path, DEVICE)
        selected_iteration = int(selected_checkpoint["iteration"])
        model.load_state_dict(selected_checkpoint["model_state"], strict=True)
        model.eval()
        print(
            "Checkpoint selection is complete. Running the full-cohort "
            f"risk-grid diagnostic once for selected iteration {selected_iteration}.",
            flush=True,
        )
        selected_discrimination_metrics, selected_discrimination_frame = (
            validation_risk_reporting_metrics(
                model,
                VAL_SAMPLES,
                iteration=int(selected_iteration),
                is_selected_checkpoint=True,
            )
        )
        selected_discrimination_frame.insert(
            0, "iteration", selected_iteration
        )
        selected_discrimination_frame.to_csv(
            RUN_DIR / "selected_checkpoint_validation_discrimination.csv",
            index=False,
        )
        atomic_json_dump(
            {
                "selected_iteration": selected_iteration,
                "checkpoint_sha256": sha256_file(best_path),
                **selected_discrimination_metrics,
            },
            RUN_DIR / "selected_checkpoint_validation_metrics.json",
        )
        for history_row in history:
            if int(history_row["iteration"]) == selected_iteration:
                history_row.update(selected_discrimination_metrics)
                break
        pd.DataFrame(history).to_csv(history_path, index=False)
        if not CFG.log_validation_metrics:
            selected_discrimination_frame.to_csv(
                discrimination_history_path,
                index=False,
            )
        print(
            "Selected-checkpoint validation risk diagnostics complete.",
            flush=True,
        )
        atomic_json_dump({
            "selection_metric": (
                "minimum original Delphi validation next-event plus waiting-time loss"
            ),
            "best_validation_original_delphi_nll": (
                best_validation_original_delphi_nll
            ),
            "training_objective": (
                "original full-vocabulary Delphi next-event cross-entropy plus "
                "competing-exponential waiting-time likelihood only"
            ),
            "landmark_auxiliary_weight": 0.0,
            "validation_discrimination_role": "diagnostic_only_not_selection",
            "validation_discrimination_schedule": (
                "selected checkpoint once after AR-NLL checkpoint selection"
                if not CFG.log_validation_metrics
                else f"every {CFG.validation_risk_report_interval:,} updates plus selected checkpoint"
            ),
            "validation_discrimination_horizons_months": list(
                CFG.validation_risk_horizons_months
            ),
            "selected_checkpoint_validation_metrics": (
                selected_discrimination_metrics
            ),
            "validation_rate_scaling_enabled": bool(CFG.enable_validation_rate_scaling),
            "test_used_for_selection": False,
            "checkpoint_sha256": sha256_file(best_path),
            "canonical_index_contract": str(CANONICAL_PATHS["input_manifest"]),
            "canonical_index_design": CANONICAL_INDEX_DESIGN,
            "completed_unix_time": time.time(),
        }, RUN_DIR / "training_complete.json")
        return best_path
    except BaseException as error:
        atomic_json_dump({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "failed_unix_time": time.time(),
        }, RUN_DIR / "training_failed.json")
        raise


def resolved_checkpoint_path() -> Path:
    return (
        Path(CFG.checkpoint_path).expanduser().resolve()
        if CFG.checkpoint_path
        else RUN_DIR / "best_checkpoint.pt"
    )


def load_frozen_model(
    checkpoint_path: Path,
) -> DelphiCanonicalModel:
    checkpoint_path = Path(checkpoint_path)
    require_files([checkpoint_path])
    checkpoint = torch_load_compat(checkpoint_path, DEVICE)
    if (
        checkpoint.get("preprocessing_contract_sha256")
        != PREPROCESSING_CONTRACT_HASH
    ):
        raise RuntimeError(
            "Checkpoint preprocessing contract does not match this run."
        )
    if (
        checkpoint.get("condition_axis_sha256")
        != canonical_json_hash(CONDITION_NAMES)
    ):
        raise RuntimeError(
            "Checkpoint condition axis does not match the canonical axis."
        )
    if (
        checkpoint.get("scientific_config_sha256")
        != canonical_json_hash(MODEL_SCIENTIFIC_CONFIG)
    ):
        raise RuntimeError(
            "Checkpoint scientific configuration differs from this notebook."
        )
    model = DelphiCanonicalModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.output_head.weight is not model.token_embedding.weight:
        raise RuntimeError("Delphi tied weights were not restored.")
    model.eval()
    return model
