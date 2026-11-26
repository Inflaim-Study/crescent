# Dynamic-DeepHit-inspired — Shared predictors, delayed-entry targets, and dataloaders
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 5 ---
def selected_history(row):
    ids=np.asarray(row.history_event_ids,np.int64)
    types=np.asarray(row.history_event_type_ids,np.int64)
    times=np.asarray(row.history_event_times_months,float)
    values=np.asarray(row.history_event_values,np.float32)
    masks=np.asarray(row.history_event_value_masks,np.float32)
    if len({len(x) for x in [ids,types,times,values,masks]})!=1:
        raise ValueError("Misaligned canonical event fields.")
    keep=~np.isin(ids,[PAD,CLS,INDEX])
    age=float(row.index_age_months)
    if not np.isfinite(age) or np.any(keep & (~np.isfinite(times) | (times>age+12./365.25+1e-6))):
        raise ValueError(f"Invalid predictor time for {row.eid}.")
    idx=np.flatnonzero(keep)
    idx=idx[np.lexsort((types[idx],ids[idx],times[idx]))]
    if len(idx)>25:
        recent=idx[-25:].copy()
        crp=idx[(types[idx]==CRP_TYPE)|np.isin(ids[idx],CRP_IDS)]
        if len(crp) and crp[-1] not in recent:
            recent[0]=crp[-1]
        idx=recent
    return (np.r_[ids[idx],CLS],np.r_[types[idx],TCLS],
            np.r_[np.minimum(times[idx],age),age].astype("f"),
            np.r_[np.nan_to_num(values[idx],nan=0.,posinf=0.,neginf=0.),np.float32(0.)],
            np.r_[(masks[idx]>=.5).astype("f"),np.float32(0.)])

def outcomes(frame):
    active=np.stack(frame.cr_at_risk).astype(bool)
    et=np.stack(frame.cr_event_type).astype(np.int8)
    t=np.stack(frame.cr_event_time_months).astype(float)
    c=frame.C_months.to_numpy(float)
    d=frame.death_time_months.to_numpy(float)
    dead=frame.death_observed_post_index.to_numpy(bool)&np.isfinite(d)
    event=np.isin(et,[1,2])&np.isfinite(t)
    if active.shape!=(len(frame),K) or not np.isfinite(c).all() or np.any(event & (t>c[:,None]+1e-4)):
        raise ValueError("Canonical time/target axis inconsistent with follow-up.")
    eligible=active & (c[:,None]>ENTRY+ENTRY_EPS) & ~(event & (t<=ENTRY+ENTRY_EPS))
    status=np.where(eligible & event, et,0).astype(np.int8)
    stop=np.where(eligible & event,t,c[:,None])
    de=(c>ENTRY+ENTRY_EPS)&~(dead&(d<=ENTRY+ENTRY_EPS))
    ds=np.where(de&dead,d,c)
    return dict(active=eligible,status=status,stop=stop,death_active=de,
                death_status=de&dead,death_stop=ds)

class Participants(Dataset):
    def __init__(self,frame):
        self.eids=frame.eid.astype(str).to_numpy()
        self.targets=outcomes(frame)
        self.history=[selected_history(r) for r in frame.itertuples(index=False)]
        self.numeric=np.stack(frame.continuous_static).astype("f")
        if not np.isfinite(self.numeric).all():
            raise ValueError("Non-finite static inputs.")
        self.cats=frame[["gender_idx","ethnicity_idx","townsend_q"]].to_numpy(np.int64).copy()
        for j,(size,unknown) in enumerate([("N_GENDER_MODEL","UNKNOWN_GENDER_IDX"),
                                           ("N_ETHNICITY_MODEL","UNKNOWN_ETHNICITY_IDX"),
                                           ("N_TOWNSEND_MODEL","UNKNOWN_TOWNSEND_Q")]):
            bad=(self.cats[:,j]<0)|(self.cats[:,j]>=int(st[size]))
            self.cats[bad,j]=int(st[unknown])
    def __len__(self):
        return len(self.history)
    def __getitem__(self,i):
        return i

def collate(indices,dataset):
    rows=[dataset.history[i] for i in indices]
    lengths=np.array([len(r[0]) for r in rows])
    shape=(len(rows),int(lengths.max()))
    x={"lengths":torch.as_tensor(lengths),"cats":torch.from_numpy(dataset.cats[indices]),
       "numeric":torch.from_numpy(dataset.numeric[indices])}
    for j,(key,pad,dtype) in enumerate([("ids",PAD,"int64"),("types",TPAD,"int64"),("time",0,"f"),
                                         ("value",0,"f"),("mask",0,"f")]):
        a=np.full(shape,pad,dtype=dtype)
        for i,r in enumerate(rows):
            a[i,:lengths[i]]=r[j]
        x[key]=torch.from_numpy(a)
    x["attention_mask"] = torch.arange(shape[1])[None,:] < x["lengths"][:,None]
    x["index_age_months"] = x["time"][torch.arange(len(rows)),x["lengths"]-1]
    y={k:torch.from_numpy(v[indices]) for k,v in dataset.targets.items()}
    return x,y

def loader(dataset,shuffle=False,generator=None):
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=shuffle,generator=generator,
                      num_workers=0,pin_memory=(DEVICE.type=="cuda"),collate_fn=partial(collate,dataset=dataset))
def device_batch(d):
    return {k:v.to(DEVICE,non_blocking=True) for k,v in d.items()}

train=Participants(train_frame)
val=Participants(val_frame)
shuffle_rng=torch.Generator().manual_seed(SEED)
train_loader=loader(train,True,shuffle_rng)
val_loader=loader(val)
