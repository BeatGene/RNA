bg_color white
set grid_mode, 0

# --- 3bbm (seed=44) ---
load 3bbm_seed44_gt.pdb, 3bbm_seed44_gt
load 3bbm_seed44_pre.pdb, 3bbm_seed44_pre
load 3bbm_seed44_post.pdb, 3bbm_seed44_post
color green, 3bbm_seed44_gt
color red, 3bbm_seed44_pre
color blue, 3bbm_seed44_post
show cartoon, 3bbm_seed44_gt
show cartoon, 3bbm_seed44_pre
show cartoon, 3bbm_seed44_post

# --- 2jsg (seed=42) ---
load 2jsg_seed42_gt.pdb, 2jsg_seed42_gt
load 2jsg_seed42_pre.pdb, 2jsg_seed42_pre
load 2jsg_seed42_post.pdb, 2jsg_seed42_post
color green, 2jsg_seed42_gt
color red, 2jsg_seed42_pre
color blue, 2jsg_seed42_post
show cartoon, 2jsg_seed42_gt
show cartoon, 2jsg_seed42_pre
show cartoon, 2jsg_seed42_post

# --- 6bfb (seed=45) ---
load 6bfb_seed45_gt.pdb, 6bfb_seed45_gt
load 6bfb_seed45_pre.pdb, 6bfb_seed45_pre
load 6bfb_seed45_post.pdb, 6bfb_seed45_post
color green, 6bfb_seed45_gt
color red, 6bfb_seed45_pre
color blue, 6bfb_seed45_post
show cartoon, 6bfb_seed45_gt
show cartoon, 6bfb_seed45_pre
show cartoon, 6bfb_seed45_post

# --- 6ufm (seed=45) ---
load 6ufm_seed45_gt.pdb, 6ufm_seed45_gt
load 6ufm_seed45_pre.pdb, 6ufm_seed45_pre
load 6ufm_seed45_post.pdb, 6ufm_seed45_post
color green, 6ufm_seed45_gt
color red, 6ufm_seed45_pre
color blue, 6ufm_seed45_post
show cartoon, 6ufm_seed45_gt
show cartoon, 6ufm_seed45_pre
show cartoon, 6ufm_seed45_post

# --- 6ge1 (seed=43) ---
load 6ge1_seed43_gt.pdb, 6ge1_seed43_gt
load 6ge1_seed43_pre.pdb, 6ge1_seed43_pre
load 6ge1_seed43_post.pdb, 6ge1_seed43_post
color green, 6ge1_seed43_gt
color red, 6ge1_seed43_pre
color blue, 6ge1_seed43_post
show cartoon, 6ge1_seed43_gt
show cartoon, 6ge1_seed43_pre
show cartoon, 6ge1_seed43_post


set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1


align 3bbm_seed44_pre, 3bbm_seed44_gt
align 3bbm_seed44_post, 3bbm_seed44_gt

align 2jsg_seed42_pre, 2jsg_seed42_gt
align 2jsg_seed42_post, 2jsg_seed42_gt

align 6bfb_seed45_pre, 6bfb_seed45_gt
align 6bfb_seed45_post, 6bfb_seed45_gt

align 6ufm_seed45_pre, 6ufm_seed45_gt
align 6ufm_seed45_post, 6ufm_seed45_gt

align 6ge1_seed43_pre, 6ge1_seed43_gt
align 6ge1_seed43_post, 6ge1_seed43_gt

disable all

# Group 1: 3bbm_seed44
group group_1, 3bbm_seed44_gt 3bbm_seed44_pre 3bbm_seed44_post

# Group 2: 2jsg_seed42
group group_2, 2jsg_seed42_gt 2jsg_seed42_pre 2jsg_seed42_post

# Group 3: 6bfb_seed45
group group_3, 6bfb_seed45_gt 6bfb_seed45_pre 6bfb_seed45_post

# Group 4: 6ufm_seed45
group group_4, 6ufm_seed45_gt 6ufm_seed45_pre 6ufm_seed45_post

# Group 5: 6ge1_seed43
group group_5, 6ge1_seed43_gt 6ge1_seed43_pre 6ge1_seed43_post

zoom all

