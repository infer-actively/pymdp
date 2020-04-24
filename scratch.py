from inferactively.core import construct_policies

pols, n_control = construct_policies([2, 5], 2, n_control=None, policy_len=2, control_fac_idx=None)
print(pols)