# Nb of neighbours and sample size

## Init: random
Higher number of samples -> More difficult to recover the circle
- $n=5000$, 13 cases not recovered
- $n=20000$, 18 cases not recovered
- $n=100000$, all cases but one!

Possible explanation:
- forcing all points to live in $[0,10]^2$ using the same value of `min_dist` and `spread` means that the more points there are, the more they fight over space. Since there are so many points in the second case, repulsion is higher and the circle is not recovered in more cases.
- they require more epochs?

### n=5000

![[report_experiment_circle_a_1_sample_5000.png]]
### n=20000
![[report_experiment_circle_a_1_sample_20000.png]]

### n=100000
![[report_experiment_circle_a_1_sample_100000.png]]


## Init: spectral

The same does not happen with spectral initialisation. It is interesting to see how in the third case, the circle shape is better seen when the number of neighbours is higher, which might mean that global structure is preserved better when this number is higher (as intuition would suggest)

### n=5000
![[report_experiment_circle_a_1_sample_5000_spectral.png]]
### n=20000
![[report_experiment_circle_a_1_sample_20000_spectral.png]]
### n=100000

![[report_experiment_circle_a_1_sample_100000_spectral.png]]



# Hypercube 0

## List of HPs
Experiment 0
	n_neighbors:	15
	min_dist:	1.482735559100664
	spread:	6.676458525412038
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 1
	n_neighbors:	31
	min_dist:	0.6675879196194421
	spread:	3.3246850561321333
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 2
	n_neighbors:	8
	min_dist:	0.11484869341094311
	spread:	1.6340521647286659
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 3
	n_neighbors:	44
	min_dist:	0.4572157103431793
	spread:	9.780432099487149
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 4
	n_neighbors:	12
	min_dist:	0.2943155430464348
	spread:	4.5958974310894085
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 5
	n_neighbors:	7
	min_dist:	3.03678753634655
	spread:	0.5069850851449073
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 6
	n_neighbors:	21
	min_dist:	1.7314598502354024
	spread:	0.5821294703935742
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 7
	n_neighbors:	37
	min_dist:	4.724729885211947
	spread:	2.383406210504778
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 8
	n_neighbors:	46
	min_dist:	3.7603078816300872
	spread:	1.1020970396179846
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 9
	n_neighbors:	27
	min_dist:	0.05904014585983891
	spread:	7.929954419359145
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 10
	n_neighbors:	33
	min_dist:	0.19790947612659465
	spread:	2.062055327756497
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 11
	n_neighbors:	39
	min_dist:	0.1256154278649532
	spread:	0.9716148080051867
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 12
	n_neighbors:	10
	min_dist:	0.9823664457039214
	spread:	1.2853273973148873
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 13
	n_neighbors:	17
	min_dist:	1.2424232436896667
	spread:	5.63322881677
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 14
	n_neighbors:	19
	min_dist:	0.20167999515508048
	spread:	0.7445028565645148
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 15
	n_neighbors:	35
	min_dist:	0.08353864666968566
	spread:	5.370835423729109
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 16
	n_neighbors:	43
	min_dist:	0.5609583529438136
	spread:	1.7564864444356507
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 17
	n_neighbors:	48
	min_dist:	0.07088033474727123
	spread:	2.645261424357481
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 18
	n_neighbors:	29
	min_dist:	2.4655719396964626
	spread:	0.8016221208489014
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12

Experiment 19
	n_neighbors:	24
	min_dist:	0.32846943225327807
	spread:	3.5514197684106517
	n_epochs:	[0, 1000, 2000, 3000, 4000, 5000]
	init:	random
	n_jobs:	12



## Init: random
### n=20000
![[report_exp_hypercube_0_sample_20000.png]]



## Init: spectral

### n=20000
![[report_exp_hypercube_0_sample_20000_spectral.png]]




# 5 dim of noise

## n=5000

### std=0.1
![[report_circle_dim_5_std_0_1_sample_5000.png]]
![[fast_tsne_plot_p_30.png]]

![[fast_tsne_plot_p_60.png]]

### std=0.5
![[report_circle_dim_5_std_0_5_sample_5000.png]]
### std=1

![[report_circle_dim_5_std_1_sample_5000.png]]
### std=4

![[report_circle_dim_5_std_4_sample_5000.png]]




# Hypercube 1

