"""Pure selection-only gate for the predeclared two-cohort IoU experiment."""
IOU_FAMILY='whole-window-soft-jaccard-v1'
OFFSETS=(101,202,303)

def iou_gate(candidates,protocol_sha):
    family=[c for c in candidates if c.get('experiment_family')==IOU_FAMILY and c.get('protocol_sha256')==protocol_sha]
    import math
    groups={};roots=[]
    for c in family:
        cohort=c.get('cohort')
        if cohort not in ('primary','replica') or cohort in groups:
            return dict(passed=False,reason='Duplicate or unexpected cohort')
        if not c.get('promotion_requires_both_cohorts') or c.get('confirmation_evaluated') is not False:
            return dict(passed=False,reason='Protocol or sealed-confirmation mismatch')
        try:
            control=[float(c['selection_results'][f'A_{s}']['sel_exact']['overall']) for s in OFFSETS]
            delta=[float(c['selection_results'][f'B_{s}']['sel_exact']['overall'])-a for s,a in zip(OFFSETS,control)]
            persistence=float(c['selection_persistence']);roots.append(int(c['root_seed']))
        except (KeyError,TypeError,ValueError):return dict(passed=False,reason='Incomplete paired seeds or persistence')
        if not all(math.isfinite(v) for v in control+delta+[persistence]):return dict(passed=False,reason='Non-finite scores')
        if not all(a-persistence>=.1 for a in control):return dict(passed=False,reason='Control failed preregistered persistence validity check')
        groups[cohort]=delta
    if set(groups)!= {'primary','replica'}:return dict(passed=False,reason='Both three-seed cohorts required')
    if len({r+s for r in roots for s in OFFSETS})!=6:return dict(passed=False,reason='Six distinct seeds required')
    values=groups['primary']+groups['replica'];means={k:sum(v)/3 for k,v in groups.items()}
    mean=sum(values)/6;positive=sum(x>0 for x in values)
    variance=sum((v-mean)**2 for v in values)/5
    # Two-sided Student t(.975,df=5), independently checked against scipy before execution.
    margin=2.570581835636305*math.sqrt(variance/6);ci=[mean-margin,mean+margin]
    passed=all(v>0 for v in means.values()) and mean>=.01 and positive>=4 and ci[0]>0
    return dict(passed=passed,reason='Fixed six-seed selection criterion',mean_delta=mean,positive_seeds=positive,
                cohort_mean_delta=means,conditional_seed_ci95=ci,
                ci_caveat='Only initialization variance on shared observed proxy data; not independent data/generalization or official-score uncertainty.',
                confirmation_evaluated=False,official_score=False)
