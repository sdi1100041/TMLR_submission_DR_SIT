Just run like the following

#2000 timesteps 20 variables, lag=2, noise to signal ratio 0
python official.py --N_data=2000 --path=DAGs/SettingTest/time_series_data/MLP/20_2/a_1_pc_0.5_nsr_0

#2000 timesteps 10 variables, lag=2, noise to signal ratio 0.05
python official.py --N_data=2000 --path=DAGs/SettingTest/time_series_data/MLP/10_2/a_1_pc_0.5_nsr_0.05

#2000 timesteps 40 variables, lag=2, noise to signal ratio 0.25
python official.py --N_data=2000 --path=DAGs/SettingTest/time_series_data/MLP/40_2/a_1_pc_0.5_nsr_0.25
