# Dynamic-DeepHit-inspired — Validation scoring, 50,000-update training, and test prediction export
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 9 ---
class CensoringKM:
    def __init__(self,stop,status):
        order=np.argsort(stop,kind="stable")
        self.times,first=np.unique(stop[order],return_index=True)
        if len(first):
            censor=np.add.reduceat((status[order]==0).astype(int),first)
            self.after=np.cumprod(1-censor/(len(stop)-first))
        else:
            self.after=np.empty(0)
    def before(self,t):
        index=np.searchsorted(self.times,t,side="left")
        return np.maximum(np.r_[1.,self.after][index],G_FLOOR)

def censoring_models(y):
    kms=[]
    for k in range(K):
        a=y["active"][:,k]
        kms.append(CensoringKM(y["stop"][a,k]-ENTRY,y["status"][a,k]))
    a=y["death_active"]
    return kms,CensoringKM(y["death_stop"][a]-ENTRY,y["death_status"][a].astype(int))

def binary_window(active,status,stop,h,km):
    t=stop-ENTRY
    case=active&(status==1)&(t<=h+TIME_EPS)
    competing=active&(status==2)&(t<=h+TIME_EPS)
    other=active&(t>=h-TIME_EPS)&~(case|competing)
    w=np.zeros(len(t))
    w[case|competing]=1/km.before(t[case|competing])
    w[other]=1/km.before(h)
    return case,w

def scores_at_horizon(risk,case,w,active):
    keep=w>0
    auc=ap=np.nan
    if case[keep].any() and (~case[keep]).any():
        auc=roc_auc_score(case[keep],risk[keep],sample_weight=w[keep])
        ap=average_precision_score(case[keep],risk[keep],sample_weight=w[keep])
    brier=np.dot(w,(case.astype(float)-risk)**2)/active.sum() if active.any() else np.nan
    return {"auc_ipcw_cr":auc,"auprc_ipcw_cr":ap,"brier_ipcw":brier,"known_n":int(keep.sum()),
            "event_n":int(case.sum()),"eligible_n":int(active.sum())}

kms,km_death=censoring_models(train.targets)

@torch.no_grad()
def evaluate(data_loader,horizons=(),full=False,attention_path=None):
    model.eval()
    sums=np.zeros(4)
    adj,unadj,death=[],[],[]
    attention_rows=[]
    for x,y in data_loader:
        rates=model(device_batch(x))
        if attention_path is not None:
            weights = rates[2].float().cpu().numpy()
            offset = len(attention_rows)
            for i, length in enumerate(x["lengths"].tolist()):
                a = weights[i,:length].astype(float)
                attention_rows.append(dict(eid=data_loader.dataset.eids[offset+i],sequence_length=length,
                    attention_entropy=float(-np.sum(a*np.log(np.maximum(a,1e-12)))),
                    max_attention=float(a.max()),cls_attention=float(a[-1])))
        sums+=torch.stack(components(rates,device_batch(y))).double().cpu().numpy()
        if horizons:
            a,u,d=risk_curves(rates,horizons)
            adj.append(a.cpu().numpy())
            if full:
                unadj.append(u.cpu().numpy())
                death.append(d.cpu().numpy())
    if not sums[1] or not sums[3]:
        raise ValueError("No eligible participants or pairs.")
    nll=sums[0]/sums[1]+.1*sums[2]/sums[3]
    if not math.isfinite(nll):
        raise FloatingPointError("Non-finite evaluation NLL.")
    arrays=[np.concatenate(v) if v else None for v in (adj,unadj,death)]
    if any(v is not None and (not np.isfinite(v).all() or v.min() < -1e-6 or v.max()>1.000001) for v in arrays):
        raise FloatingPointError("Invalid risk probabilities.")
    if attention_path is not None:
        save_csv(pd.DataFrame(attention_rows),attention_path)
    return nll,*arrays

def condition_scores(dataset,adj,horizons):
    y=dataset.targets
    rows=[]
    for k,name in enumerate(names):
        a=y["active"][:,k]
        for j,h in enumerate(horizons):
            case,w=binary_window(a,y["status"][:,k],y["stop"][:,k],h,kms[k])
            metrics=scores_at_horizon(adj[:,k,j].astype(float),case,w,a)
            rows.append(dict(condition_id=k,condition_name=name,horizon_months=h,risk_type="death_adjusted",
                             auroc=metrics["auc_ipcw_cr"],**metrics))
    return pd.DataFrame(rows)

def report_validation(step,adj,val_loss,restored=False):
    long=condition_scores(val,adj,POINT)
    wide=long.pivot(index=["condition_id","condition_name"],columns="horizon_months",values=["auc_ipcw_cr","event_n"])
    wide.columns=[f"auc_{h}m" if key=="auc_ipcw_cr" else f"n_cases_{h}m" for key,h in wide.columns]
    scores=wide.reset_index()
    scores["auc_mean_24_60"]=scores[["auc_24m","auc_60m"]].mean(axis=1,skipna=False)
    ok=scores["n_cases_24m"].ge(20)&scores["n_cases_60m"].ge(20)&scores["auc_mean_24_60"].notna()
    rank=scores[ok].sort_values(["auc_mean_24_60","condition_id"],ascending=[False,True])
    groups=pd.concat([rank.head(5).assign(group="top"),rank.tail(5).sort_values("auc_mean_24_60").assign(group="bottom")],ignore_index=True)
    summary={f"val_macro_auc_{h}m":float(scores[f"auc_{h}m"].mean()) for h in POINT}
    summary.update({f"val_n_conditions_{h}m":int(scores[f"auc_{h}m"].notna().sum()) for h in POINT})
    groups.insert(1,"rank",groups.groupby("group").cumcount()+1)
    if restored:
        save_csv(scores,RUN_DIR/"validation_per_condition_auroc_24_60m.csv")
        save_csv(groups,RUN_DIR/"validation_top5_bottom5_conditions_24_60m.csv")
        return {**summary, "n_rankable_conditions": len(rank)}
    save_csv(scores,RUN_DIR/f"validation_per_condition_auroc_iteration_{step:06d}.csv")
    save_csv(groups,RUN_DIR/f"validation_top5_bottom5_iteration_{step:06d}.csv")
    log(f"UPDATE {step:,}/50,000 | FULL VALIDATION n={len(val):,} | NLL {val_loss:.6f} | "
        f"24m IPCW macro AUROC {summary['val_macro_auc_24m']:.4f} ({summary['val_n_conditions_24m']} conditions) | "
        f"60m IPCW macro AUROC {summary['val_macro_auc_60m']:.4f} ({summary['val_n_conditions_60m']} conditions)")
    log(groups[["group","condition_name","auc_24m","auc_60m","auc_mean_24_60"]].to_string(index=False) if len(groups)
        else "No condition met the 20-case ranking threshold at both horizons.")
    return summary

# --- Notebook cell 11 ---
def learning_rate(step):
    if step<=WARMUP:
        return LR*step/WARMUP
    return LR_FLOOR+(LR-LR_FLOOR)*.5*(1+math.cos(math.pi*(step-WARMUP)/(MAX_UPDATES-WARMUP)))

def optimizer_for(m):
    return torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)

def load_checked(path,kind):
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state.get("contract")!=contract or state.get("kind")!=kind:
        raise ValueError(f"Checkpoint contract differs from this model or cohort: {path}")
    return state

def train_model():
    opt=optimizer_for(model)
    start,cursor,best,history=0,0,float("inf"),[]
    epoch_rng=shuffle_rng.get_state()
    if RESUME:
        state=load_checked(LAST,"resume")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start,cursor,best,history=state["step"],state["cursor"],state["best"],state["history"]
        epoch_rng=state["epoch_rng"]
        shuffle_rng.set_state(epoch_rng)
    iterator=iter(train_loader)
    if RESUME:
        for _ in range(cursor):
            next(iterator)
        torch.set_rng_state(state["cpu_rng"])
        if DEVICE.type=="cuda":
            torch.cuda.set_rng_state(state["cuda_rng"],DEVICE)
        del state
    iteration_path=RUN_DIR/"iteration_log_every_100.csv"
    iterations=[]
    if RESUME and iteration_path.is_file():
        previous=pd.read_csv(iteration_path)
        iterations=previous.loc[previous["iteration"]<=start].to_dict("records")
    window=[]
    began=time.monotonic()
    for step in range(start+1,MAX_UPDATES+1):
        try:
            x,y=next(iterator)
        except StopIteration:
            epoch_rng=shuffle_rng.get_state()
            iterator=iter(train_loader)
            cursor=0
            x,y=next(iterator)
        cursor+=1
        model.train()
        lr=learning_rate(step)
        for group in opt.param_groups:
            group["lr"]=lr
        opt.zero_grad(set_to_none=True)
        dn,nd,mn,nm=components(model(device_batch(x)),device_batch(y))
        loss=dn/nd.clamp_min(1)+.1*mn/nm.clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at {step}.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        window.append(torch.stack((loss.detach(),(dn/nd.clamp_min(1)).detach(),(mn/nm.clamp_min(1)).detach())))
        if step%100==0:
            joint,disease,mortality=torch.stack(window).mean(0).cpu().tolist()
            window.clear()
            iterations.append(dict(iteration=step,loss=joint,disease_cr_nll=disease,death_nll=mortality,
                                   lr=lr,elapsed_minutes=(time.monotonic()-began)/60))
            save_csv(pd.DataFrame(iterations),iteration_path)
            log(f"Update {step:,}/{MAX_UPDATES:,} | training NLL {loss.item():.6f} | LR {lr:.3g} | elapsed {(time.monotonic()-began)/60:.1f} min")
        if step%LOSS_INTERVAL:
            continue
        perf=(step%REPORT_INTERVAL==0)
        val_loss,adj,_,_=evaluate(val_loader,POINT if perf else ())
        row={"step":step,"lr":lr,"val_loss":float(val_loss)}
        log(f"Update {step:,} | validation NLL {val_loss:.6f}")
        if perf:
            row.update(report_validation(step,adj,val_loss))
        if val_loss<best-float(settings.get("MIN_DELTA",1e-4)):
            best=float(val_loss)
            save_pt({"kind":"best","contract":contract,"model":model.state_dict(),"step":step,"val_loss":best},BEST)
        history.append(row)
        save_pt({"kind":"resume","contract":contract,"model":model.state_dict(),"optimizer":opt.state_dict(),
                 "step":step,"cursor":cursor,"epoch_rng":epoch_rng,"best":best,"history":history,
                 "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(DEVICE) if DEVICE.type=="cuda" else None},LAST)
        save_csv(pd.DataFrame(history),RUN_DIR/"training_history.csv")
    if [r["step"] for r in history if "val_macro_auc_24m" in r]!=list(range(REPORT_INTERVAL,MAX_UPDATES+1,REPORT_INTERVAL)):
        raise RuntimeError("Missing a scheduled full-validation performance report.")
    save_csv(pd.DataFrame(history),RUN_DIR/"training_history.csv")

if WORKFLOW=="train_and_evaluate":
    train_model()
last_alias=RUN_DIR/"last_model.pt"
if LAST.exists() and not last_alias.exists() and not last_alias.is_symlink():
    last_alias.symlink_to(LAST.name)
selected=load_checked(BEST,"best")
model.load_state_dict(selected["model"],strict=True)
log(f"Restored update {selected['step']:,} | validation NLL {selected['val_loss']:.6f}")
val_nll, val_adj, _, _ = evaluate(val_loader, POINT)
restored_metrics = report_validation(selected["step"], val_adj, val_nll, restored=True)
training_summary = {
    "model_name": "Dynamic-DeepHit", "training_unit": "optimizer_iteration",
    "max_iterations_requested": MAX_UPDATES, "batch_size": BATCH_SIZE,
    "best_loss_iteration": selected["step"], "best_val_loss": selected["val_loss"],
    "primary_checkpoint_selection": "joint_nll", "restored_checkpoint_path": str(BEST),
    "weights_used_for_validation_and_checkpoint": "raw", "seed": SEED,
    "validation_macro_auroc_restored_checkpoint": restored_metrics,
    "validation_participants": len(val), "training_history_path": str(RUN_DIR / "training_history.csv")}
(RUN_DIR / "cell6_training_summary_death_adjusted.json").write_text(json.dumps(training_summary, indent=2))
del selected, val_adj

# --- Notebook cell 13 ---
test_frame=load_frame("test")
if dev_ids.intersection(test_frame.eid.astype(str)):
    raise ValueError("Overlapping test and development participants.")
test=Participants(test_frame)
def input_coverage(frame, dataset, split):
    counts, selected = [], []
    landmark = available_crp = used_crp = numeric = with_crp = dropped = 0
    for row, history in zip(frame.itertuples(index=False), dataset.history, strict=True):
        raw = np.asarray(row.history_event_ids)
        typs = np.asarray(row.history_event_type_ids)
        genuine = ~np.isin(raw, [PAD, CLS, INDEX])
        before = genuine & ((typs == CRP_TYPE) | np.isin(raw, CRP_IDS))
        ids, types, times, _, masks = history
        after = ((types[:-1] == CRP_TYPE) | np.isin(ids[:-1], CRP_IDS))
        if np.any(times > np.float32(row.index_age_months)) or np.isin(ids[:-1], [PAD, CLS, INDEX]).any():
            raise ValueError(f"Unexpected clinical inputs for {split} participant {row.eid}")
        counts.append(genuine.sum())
        selected.append(len(ids)-1)
        landmark += int((raw == INDEX).sum())
        available_crp += int(before.sum())
        used_crp += int(after.sum())
        numeric += int((after & (masks[:-1] >= .5)).sum())
        with_crp += int(after.any())
        dropped += int(before.any() and not after.any())
    n, used = np.asarray(counts), np.asarray(selected)
    return dict(split=split, participants=len(n), available_events_total=int(n.sum()),
                available_events_mean=float(n.mean()), available_events_median=float(np.median(n)),
                available_events_p90=float(np.quantile(n,.9)), available_events_p99=float(np.quantile(n,.99)),
                available_events_max=int(n.max()), participants_with_zero_genuine_events=int((n==0).sum()),
                input_events_total=int(used.sum()), available_events_outside_input=int((n-used).sum()),
                participants_exceeding_input_limit=int((n>25).sum()), artificial_landmark_tokens_in_canonical_rows=landmark,
                artificial_landmark_tokens_used_as_clinical_inputs=0, genuine_post_landmark_inputs=0,
                crp_events_available=available_crp, crp_events_in_input=used_crp, crp_events_with_numeric_value=numeric,
                crp_events_with_missing_value=used_crp-numeric, participants_with_crp_input=with_crp,
                participants_dropped_for_crp=dropped)
coverage = pd.DataFrame([input_coverage(frame, ds, split) for frame,ds,split in
                        ((train_frame,train,"train"),(val_frame,val,"val"),(test_frame,test,"test"))])
save_csv(coverage,RUN_DIR/"cell4.5_full_history_coverage_by_split.csv")
save_csv(coverage[["split","participants",*[c for c in coverage if "crp" in c]]],RUN_DIR/"crp_input_audit.csv")
loader_meta = dict(dataset_class="Participants", collate="collate", K=K, BATCH_SIZE=BATCH_SIZE,
                   full_sequence_max_events=25, full_sequence_max_len_with_cls=26,
                   post_landmark_predictors_allowed=False, history_coverage=coverage.to_dict("records"))
for filename in ("cell4.5_full_history_dataloader_metadata.json","cell4.5_dataloader_metadata_death_adjusted.json"):
    (RUN_DIR/filename).write_text(json.dumps(loader_meta,indent=2))

test_loader=loader(test)
test_loss,adj,unadj,death=evaluate(test_loader,DENSE,full=True,attention_path=RUN_DIR/"test_temporal_attention_summary.csv")
y=test.targets
arrays={"eid":test.eids.astype(str),"condition_names":np.asarray(names),"horizons_months":np.array(HORIZONS),
        "entry_delay_months":ENTRY,"index_age_years":test_frame.index_age_years.to_numpy(),
        "C_months":test_frame.C_months.to_numpy(np.float32),
        "death_time_months":test_frame.death_time_months.to_numpy(np.float32),
        "death_observed_post_index":test_frame.death_observed_post_index.to_numpy()}
for key in ("at_risk","event_obs","t_tilde"):
    arrays[key]=np.stack(test_frame[key])
for h in HORIZONS:
    j=DENSE.index(h)
    arrays[f"risk_death_adjusted_h{h}"]=adj[:,:,j]
    arrays[f"risk_unadjusted_h{h}"]=unadj[:,:,j]
    arrays[f"risk_death_h{h}"]=death[:,j]
np.savez_compressed(RUN_DIR/"test_prediction_arrays_death_adjusted.npz",**arrays)
np.savez_compressed(RUN_DIR/"test_dense_prediction_arrays_for_integrated_metrics.npz",eid=test.eids.astype(str),
                    condition_names=np.asarray(names),dense_horizons_months=np.array(DENSE),
                    entry_delay_months=ENTRY,risk_death_adjusted=adj)
log(f"Saved locked Dynamic-DeepHit predictions; test NLL {test_loss:.6f}.")
del adj, unadj, death, arrays, test, test_frame, test_loader, train, val, train_loader, val_loader, train_frame, val_frame, model
