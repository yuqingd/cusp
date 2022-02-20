from utils.utils import *
from envs.SelfPlay import ALICE, BOB
from utils.logger import Logger
import torch

def play_asp(env, trainer, goal_trainer, goals_arr, step, args, device, L):
    # Have Alice set the goals
    alice_successes = []
    bob_successes = []
    for i in range(args.num_goals):
        if goal_trainer is not None:
            #get PPO alice to generate a goal
            alice_success, final_obs = goal_trainer.generate_traj(ALICE)
            goal = torch.tensor(final_obs['xyz_state'])
        else:
            #get SAC alice to generate a goal
            alice_success, final_obs = trainer.generate_traj(ALICE)
        
        goal = torch.tensor(final_obs['xyz_state'])

        # evaluate Bob on goal
        bob_success = trainer.generate_traj(BOB, goal)
        goals_arr.append(goal.unsqueeze(0).cpu().numpy())

        bob_reward = trainer.get_bob_reward()
        if goal_trainer is not None:
            # SAC Bob specific relabelling and BC
            goal_trainer.relabel_data(goal, bob_success, bob_reward) # relabel rewards for alice and goal for BC
            if args.agent_algo == 'sac' and not bob_success:
                trainer.imitate(goal_trainer.bob_bc_rollouts) # Transfer relabelled trajectories from Alice to Bob's trainer
        else:
            trainer.relabel_data(goal, bob_success, bob_reward) # relabel rewards for alice and goal for BC
        alice_successes.append(alice_success.mean())

        bob_successes.append(bob_success.mean())
        if goal_trainer is not None:
            goal_trainer.to(device)
            goal_trainer.update(step)

        print("{} -  Alice Success: {}".format(step,  1-np.mean(bob_successes)))
        L.log('train/alice/success', 1-np.mean(bob_successes), step) #log the modified reward for alice
        print("{} - Bob success: {} ".format(step,  np.mean(bob_successes)))
        L.log('train/bob/success', np.mean(bob_successes), step) 
        print("Bob Reward: {}".format(torch.sum(torch.FloatTensor(trainer.bob_rewards), dim=0).mean()))
        L.log('train/bob/reward', torch.sum(torch.FloatTensor(trainer.bob_rewards), dim=0).mean(), step) 
            

def play_goalgan(env, trainer, goal_trainer, goals_arr, step, args, device, L):
    #smart init with feasible goals to pretrain gan
    if step == 0 or not dict(goal_trainer.cur_goals):
        print("Prefind feasible goals")
        feasible_goals = trainer.generate_initial_goals()
        dis_loss, gen_loss = goal_trainer.goal_agent.pretrain(states=feasible_goals, logger=L, outer_iters=250)
        print("Pretrain Loss of Gen and Dis: ", gen_loss, dis_loss)
        goals = goal_trainer.generate_goal()
    else:
        if step % 500 != 0:
            bob_successes = []
            for i in range(args.num_goals):
                goal = random.sample(list(goal_trainer.cur_goals.keys()), 1)[0]
                bob_success = trainer.generate_traj(BOB, list(goal)) #no alice in gan formulation
                goal_trainer.cur_goals[goal].append(bob_success)

                goals_arr.append(goal[0].detach().cpu().numpy())
                bob_successes.append(bob_success.cpu().numpy())
            L.log('train/bob/success', np.mean(bob_successes), step) #log the modified reward for alice
            print("{} - Bob success: {} ".format(step,   np.mean(bob_successes)))
        else: #update gan every 500 episodes
            goal_trainer.update_gan()
            goals = goal_trainer.generate_goal()


def play_uniform(env, trainer, goal_trainer, goals_arr, step, args, device, L):
    num_proc = args.num_processes
    goal_space = env.env_method('goal_space')[0]

    # Generate goals
    goal_dim = goal_space.shape[-1]
    goals = torch.FloatTensor(np.random.uniform(goal_space.low, goal_space.high, size=(args.num_goals, num_proc, args.num_subgoals * goal_dim)))
    print("Goal", goals)

    alice_utilities = []
    bob_utilities = []
    alice_successes = []
    bob_successes = []
    
    for goal in goals:
        #evaluate alice on goal
        alice_success = trainer.generate_traj(ALICE, goal)
        #evaluate bob on goal
        bob_success = trainer.generate_traj(BOB, goal)

        alice_utility, bob_utility = trainer.compute_utilities(args.gamma, step)
            
        alice_utilities.append(alice_utility.cpu())
        bob_utilities.append(bob_utility.cpu())

        alice_successes.append(alice_success.mean())
        bob_successes.append(bob_success.mean())

    # TODO: clean this up
    if len(goals) > 1:
        alice_utilities = torch.stack(alice_utilities)
        bob_utilities = torch.stack(bob_utilities)
        alice_successes = torch.stack(alice_successes)
        bob_successes = torch.stack(bob_successes)
    else:
        alice_utilities = torch.FloatTensor(alice_utilities)
        bob_utilities = torch.FloatTensor(bob_utilities)
        alice_successes = torch.FloatTensor(alice_successes)
        bob_successes = torch.FloatTensor(bob_successes)

    L.log('train/alice/utility', torch.mean(alice_utilities), step)
    L.log('train/bob/utility', torch.mean(bob_utilities), step) 
        
    print("{} -  Alice Success: {}".format(step,  torch.mean(alice_successes)))
    print("{} - Bob success: {} ".format(step,  torch.mean(bob_successes)))
    L.log('train/alice/success', torch.mean(alice_successes), step)
    L.log('train/bob/success', torch.mean(bob_successes), step) 
    
    # Save goals for plotting
    goals = goals.detach().cpu().numpy()
    for goal in goals:
        goals_arr.append(goal)
        
