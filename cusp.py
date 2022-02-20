from utils.utils import *
from envs.SelfPlay import ALICE, BOB
from utils.logger import Logger
import torch

def play_cusp(env, trainer, goal_trainer, goals_arr, step, args, device, L):
    num_proc = args.num_processes
    goal_space = env.env_method('goal_space')[0]
    # Generate goals
    goals = goal_trainer.generate_goal()
    
    print("Goal", goals)

    alice_utilities = []
    bob_utilities = []
    alice_successes = []
    bob_successes = []

    if args.symmetrize:
        # give easier goal first, then harder goal (goal generation order is alice friendly goal gen first, then bob friendly)
        for i, goal in enumerate(goals):
            alice_success = trainer.generate_traj(ALICE, goal)
            alice_utility = trainer.compute_agent_utilities(args.gamma, ALICE, step + i)

            alice_utilities.append(alice_utility.cpu())
            alice_successes.append(alice_success.mean())

        for i, goal in enumerate(torch.flip(goals, [0])):
            bob_success = trainer.generate_traj(BOB, goal)
            bob_utility = trainer.compute_agent_utilities(args.gamma, BOB, step + i)

            bob_utilities.append(bob_utility.cpu())
            bob_successes.append(bob_success.mean())
        
        # to compute regret, flip bob utilities/successes to match alice goal order
        bob_utilities = bob_utilities[::-1]
        bob_successes = bob_successes[::-1]

    else:
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
    goal_trainer.update(goals, alice_utilities, bob_utilities)
    regrets = []

    # Save goals for plotting
    goals = goals.detach().cpu().numpy()
    for goal in goals:
        goals_arr.append(goal)
        
    return 
