
## Q1: Behavioral Cloning

### 1.2 

For each environment, run the respective commands:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name half_cheetah --n_iter 1 --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name hopper --n_iter 1 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Walker2d.pkl --env_name Walker2d-v2 --exp_name walker2d --n_iter 1 --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl --video_log_freq -1 --eval_batch_size 5000

```

### 1.3 

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Walker2d.pkl --env_name Walker2d-v2 --exp_name walker2d --n_iter 1 --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl --video_log_freq -1 --eval_batch_size 5000
```

### 1.4

Code for Plotting the plot. 
```python
import matplotlib.pyplot as plt
import numpy as np

# Given data
means = [1965.9, 2159.3]
variance = [887.9, 512.4]

# Setting up positions and width for the bars
ind = np.arange(len(means))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

# Creating bars
rects1 = ax.bar(ind - width/2, means, width, label='Eval Mean', color='b')
rects2 = ax.bar(ind + width/2, variance, width, label='Eval STD', color='r')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_xlabel('Ant-v2 with 4 and 6 layers')
ax.set_title('Effect of Number of layers on Mean and STD')
ax.set_xticks(ind)
ax.set_xticklabels(['4 layers', '6 layers'])
ax.legend()

# Auto-labeling function
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

```


## Q2: Dagger

### 2.2  

For ant environment, run the following command:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000
```

For Hopper environment, run the following command:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name dagger_Hopper --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 --eval_batch_size 5000
```

plotting code 
```python
## For Ant-v2
Eval_AverageReturn = [1965.9271240234375, 3897.038818359375, 4481.1240234375, 4659.1533203125, 4747.1982421875, 4049.230712890625, 4799.9912109375, 4676.2294921875, 4796.06787109375, 4677.0224609375]
Eval_StdReturn = [887.9882202148438, 1032, 98.99880981445312, 107.3726806640625, 30.41508674621582, 1477, 68.9958724975586, 94.02704620361328, 50.40815734863281, 80.68122863769531]

## For Hopper-v2
Eval_AverageReturn = [ 574.4583129882812, 1586.4898681640625, 1866.7156982421875, 3525.723876953125, 3778.185546875, 3777.882080078125, 3776.879638671875, 3782.64111328125, 3784.39453125, 3776.018798828125]
Eval_StdReturn = [344.87017822265625, 399.4476318359375, 332.92352294921875, 533.2819213867188, 3.942289113998413, 2.6907248497009277, 2.5862035751342773, 16.65606689453125, 2.953282356262207, 2.7762680053710938]


# write code to plot the results with error bars
# HINT: you can use matplotlib's errorbar function [OK]
import matplotlib.pyplot as plt
import numpy as np
plt.errorbar(np.arange(10), Eval_AverageReturn, yerr=Eval_StdReturn, fmt='o')

# plot a line along the average returns
plt.plot(np.arange(10), Eval_AverageReturn)
# plt.plot(np.arange(10), Eval_AverageReturn, 'o')
plt.xlabel('Iterations')
plt.ylabel('Mean and STD with Dagger')
plt.legend(['Eval_AverageReturn', 'Eval_StdReturn'])
plt.title('Dagger performance on Ant-v2 over 10 iterations')

plt.show()
```
