from utils.utils import *
from envs.registry import make_vec_envs
from envs.SelfPlay import ALICE, BOB
from utils.logger import Logger
import torch
import os
import re
from agents.sac_trainer import SACTrainer
from agents.goal_generator_trainer import GoalGeneratorTrainer
import hydra
from utils.plot_goals import plot
from cusp import play_cusp
from baselines import play_asp, play_goalgan, play_uniform
from agents.ppo_trainer import PPOTrainer

import submitit

def main():
    c = []
    hydra.main(config_path="config/train.yaml")(c.append)()
    args = c[0]
    
    if not args.no_cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.submit:
        executor = submitit.AutoExecutor(
            folder=os.path.join(args.logdir, "slurm"), slurm_max_num_timeout=3
        )
        executor.update_parameters(
            name=args.exp_name,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(args.num_workers + 1),
            tasks_per_node=1,
            nodes=args.num_nodes
        )
        job = executor.submit(Runner(), args)
        print('Submitted job:', job.job_id)
    else:
        Runner()(args)

class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None
        self.chkpt_path = None

    def __call__(self, args):
        print(args.pretty())
        torch.autograd.set_detect_anomaly(True)

        # CUDA setup 
        if not args.no_cuda and torch.cuda.is_available():
            device_str = "cuda:0"
            os.environ['MUJOCO_GL'] = 'egl'
        else:
            device_str = "cpu"
        
        device = torch.device(device_str)
        set_seed_everywhere(args.seed)

        # Logging
        log_dir = os.path.join(os.getcwd(), args.logdir)
        model_save_path = os.path.join(log_dir, "models")

        reload_model = False
        if os.path.isdir(model_save_path):
            # if model folder exists already, reload previous models and continue run
            reload_model = len(os.listdir(model_save_path)) > 0  
        if not reload_model:
            cleanup_log_dir(log_dir)
            cleanup_log_dir(model_save_path)

        L = Logger(log_dir,
            save_tb=args.log_save_tb,
            log_frequency=args.log_frequency,
            agent=args.agent.name)

        # Envs 
        env_name = args.env_name
        envs = make_vec_envs(env_name=args.env_name, 
                             seed=args.seed, 
                             log_dir=log_dir, 
                             num_processes=args.num_processes, 
                             max_episode_steps=args.num_steps_bob, 
                             num_subgoals=args.num_subgoals,
                             additional_dims=args.infeasible_dims)
        eval_envs = make_vec_envs(env_name=args.env_name, 
                                 seed=args.seed, 
                                 log_dir=log_dir, 
                                 num_processes=args.num_processes, 
                                 max_episode_steps=args.num_steps_bob, 
                                 num_subgoals=args.num_subgoals,
                                 additional_dims=args.infeasible_dims)
        eval_rng = np.random.default_rng(args.seed)
        envs.env_method('set_device', device)
        
        # Initialize learner trainer
        trainer = SACTrainer(args, L, envs, device_str, log_dir, model_save_path, reload_model)
        if args.agent_algo == 'ppo':
            trainer = PPOTrainer(args, L, envs, device_str, log_dir, model_save_path, reload_model)
        
        # Initialize goal generator trainer
        if args.goal_algo in ['cusp', 'goalgan']:   
            goal_trainer = GoalGeneratorTrainer(args, L, envs, device, model_save_path, reload_model)
        elif args.goal_algo == 'asp':
            goal_trainer = PPOTrainer(args, L, envs, device, log_dir, model_save_path, reload_model)
        else:
            goal_trainer = None

        goals_arr = []
        regrets = []
        if reload_model and os.path.exists(log_dir + '/goals.npy'):
            goals_arr = list(np.load(log_dir + '/goals.npy'))

        # Run episodes
        for step in range(trainer.episode_counter, int(args.num_steps)):
            if step % args.eval_every == 0 and step != 0:
                with torch.no_grad():
                    trainer.evaluate(eval_envs, eval_rng, step, is_alice=False) 
                    if args.eval_alice:
                        trainer.evaluate(eval_envs, eval_rng, step, is_alice=True) 

                # Update regret replay buffer
                if step > args.before_update_stale_regrets:
                    goal_trainer.check_regrets(trainer.alice, trainer.bob)

            print('Round:', step)
            if args.goal_algo == 'cusp':
                play_cusp(envs, trainer, goal_trainer, goals_arr, step, args, device, L)
            elif args.goal_algo == 'asp':
                play_asp(envs, trainer, goal_trainer, goals_arr, step, args, device, L)
            elif args.goal_algo == 'goalgan':
                play_goalgan(envs, trainer, goal_trainer, goals_arr, step, args, device, L)
            elif args.goal_algo == 'dr':
                play_uniform(envs, trainer, goal_trainer, goals_arr, step, args, device, L)

            if step % args.save_every == 0 and step > 0: 
                trainer.save(model_save_path, step)
                if goal_trainer is not None:
                    goal_trainer.save(model_save_path, step)

                # Plot goals
                if len(goals_arr) > 0:
                    with open(log_dir + '/goals.npy', 'wb') as f:
                        np.save(f, goals_arr)
                        plot(goals_arr, log_dir + '/goals_plot_' + str(step) + '.png',
                            contour=args.dummy_env, perturb=step*args.moving_regret_coeff)
        
        envs.close()
        with open(log_dir + '/goals.npy', 'wb') as f:
            np.save(f, goals_arr)
        plot(goals_arr, log_dir + '/goals_plot_' + str(step) + '.png',
                            contour=args.dummy_env, perturb=step*args.moving_regret_coeff)

if __name__ == '__main__':
    main()
