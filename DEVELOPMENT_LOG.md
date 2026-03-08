
TWIST TRAINING:        

2026-01-31_14-23-28_twist_training：将奖励向上的奖励设为0，下落过程中机器人完全没有任何动作。


2026-01-31_15-19-32_twist_training: 希望能够用腿来做缓冲，把之前训练的掏出来了，把向上奖励大大增加


2026-01-31_16-13-04_twist_training: 和之前的配置完全一样，只修改了向上奖励

2026-01-31_23-37-30_twist_training: 不小心开错了，跟之前配置一样

2.2
2026-02-02_15-57-37_twist_training：用“除缩放以外其余与gallop一样”那个点的参数来跑twist，希望能学会脚撑地的帅气动作————————————“练出来尽然是翘一只脚跑的”







GALLOP TRAINING:

2026-01-31_14-53-06_resum--new_start_easy_gait_reward--real_gallop_gait_reward_big_weight：resum easy gait，加大了步态奖励，将别的限制变低，甚至取消了直立奖励。————————“到了4m/s直接扑该”
2026-02-01_14-05-33_resum--new_start_easy_gait_reward--bigger_real_gait_reward: 上面配置不变，加了向上奖励。————————“一坨屎，不升级，训练后瞧一只脚跑”



2026-01-31_19-55-40_new_start_feet_gait_offset: 修改了feet_gait函数，新增了脚步offset，重新开始训练。————————————“没给向上奖励，直接趴着了”
2026-01-31_20-58-54_new_start_feet_gait_offset：除了向上奖励其他的跟上一次一样。————————“能跑，前腿还行，后腿给的不利索”



2026-01-31_23-05-57_resum--new_start_easy_gait_reward--no_gait：拿bound练出来的不给gait，让他随意发挥。————————————“hip张的特别开，趴着跑的，脚步并没有踏地时间差”



2026-02-01_12-32-51_bug_gait_small_gap：  修改一下落地时序看看效果。————————“像一只蛆一样在地上瞎几把乱拱”
2026-02-01_15-41-21_bug_gait2： 完全按照之前的2026-01-19_13-11-48_new_start_dynamic_gait3的配置，只改了时间方差和gait相位。————————————“差不多三只脚着地来跑”



也许低速就并不适合gallop，直接用gallop再已有训练上叠加
1.resum_bug_gait2:  2026-02-01_18-26-58_resum--new_start_easy_gait_reward--more_training。————————“抬着一只脚跑的”  

2.用之前的bound来resum offset gait:   2026-02-01_18-44-28_resum--new_start_easy_gait_reward--offset_gait。————————“两只脚差距不大，基本同时落地”  ——————>用bound跑过以后很难纠正的过来




3.用feet-gait，把std给大点，把互斥关了


4.重新弄一个更合理的offset：  2026-02-02_14-48-25_new_gallop_reward：直接让claude4.5opus写了一个，没看，直接用



2026-02-01_22-59-07_no_gait_Emergence： 也没有加gait，只不过其他奖励随便配的。————————“步态差不多是trot，一只脚腾空”
2026-02-01_23-53-23_no_gait_Emergence2:  用new_start_easy_gait的参数去掉gait reward。————————“步态差不多是gallop”
————————> 靠emergence是可以练出gallop的，不过非常heuristic

2026-02-02_11-58-33_resum--no_gait_Emergence2--more_training： 减少限制，继续训练。————————“脚步方差控制的不好，前脚还行，后脚有一只脚时长会腾空”


发现近期训练的东西都有一个共同的问题，步频太高了，可能是和动作变化率有关，把“按之前成功的gallop改版”的参数拿过来重新训练试试，同时将pd参数也设置成和之前一模一样了：2026-02-03_10-35-24_old_real_gallop。————————“一坨屎，全删了”


2026-02-03_13-19-00_GallopGait_reward:  重修改了一版gallopgaitreward，应该是最连续最吊的gallopreward了。训练配置里面给的动作变化率惩罚比较大，加速度惩罚也给。————————“效果可以”

2026-02-03_13-23-31_GallopGait_reward_less_Constraint:  较上一个配置的限制放宽了一些。————————“坐在地上走的，很奇怪”


2026-02-03_15-55-57_resum--GallopGait_reward--more_training:  继续训练。————————“效果可以，就是用的transverse gallop，前脚时差好像比后脚小”

2026-02-03_18-40-50_resum--GallopGait_reward--more_training--spine_reward： 加脊柱reward继续训。————————“脊柱基本没动”






2026-02-03_23-36-19_GallopGait_reward_rotary_gallop：   改了一下reward，用rotary gallop重修训。————————“还可以，脊柱有略微动”
2026-02-04_09-50-59_resum--GallopGait_reward_rotary_gallop--spine_training：    继续直接加脊柱reward训。————————“效果不好，不仅脊柱没动，腿还变形了，前腿更加同步，后腿非常大的不同步”

2026-02-04_23-59-51_resum--GallopGait_reward_rotary_gallop--spine_training2:   resum rotary galllop,把spine reward换成后脚的。————————“脊柱动作确实很大，但基本上是后半身在动，前半身没动，后腿基本同步，前腿同步性变的糟糕，”

2026-02-05_14-08-08_resum--resum--GallopGait_reward_rotary_gallop--spine_training3：    前脚的目标时间差给的小了些。————————“前脚同步性还是差”

spine reward会影响脚的同步性

2026-02-05_17-30-27_resum--resum--GallopGait_reward_rotary_gallop--spine_training4:     前脚时间差再给小，脊柱奖励改为0.8，速度奖励6


2026-02-05_17-53-09_resum--resum--GallopGait_reward_rotary_gallop--spine_training_average_spine_reward:   脊柱奖励改为平均脊柱奖励




bias3:    [-0.6,-0.1]
bias2.7:  [-0.25,0.4]
bias2.5:  [-0.28,0.28]
bias2.6:  [-0.8,-0.1]



-0.13，0.18



TURNING TRAINING:
2026-02-04_10-19-33_Turning_training:  从零开始带角速度训练。————————“效果很好，6rad/s随便跑”
2026-02-04_10-54-05_resum--GallopGait_reward_rotary_gallop--turning_training：  rotary gallop基础上训的。————————“效果不如直接从零开始训”







1.设计锁死脊柱实验
2.isaac里面录demo
3.mujoco录demo