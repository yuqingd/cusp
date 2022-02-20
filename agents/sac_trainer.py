import torch
import os
from agents.replay_buffer import ReplayBuffer, BCReplayBuffer
import utils.utils as utils
from utils.utils import VideoRecorder
import time
import numpy as np
from envs.SelfPlay import ALICE, BOB, get_test_cases, generate_eval_goals
import hydra
import re
import os
import matplotlib.pyplot as plt
import pickle


class SACTrainer:
    def __init__(self, args, L, envs, device, log_dir, model_save_path, reload_model=False):
        self.args = args
        self.logger = L
        self.device = torch.device(device)
        self.env = envs
        self.alice_step = 0
        self.bob_step = 0
        self.init_obs = None
        self.episode_counter = -1

        self.eval_scores = {}
        self.log_dir = log_dir

        args.agent.params.obs_dim = self.env.observation_space["full_state_with_goal"].shape[0]
        args.agent.params.goal_dim = 0


        args.agent.params.action_dim = self.env.action_space.shape[0]
        args.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        args.agent.params.device = device
        args.agent.params.avec_loss = args.avec_loss
        self.alice = hydra.utils.instantiate(args.agent)
        self.bob = hydra.utils.instantiate(args.agent)

        if reload_model:
            for checkpoint in os.listdir(model_save_path):
                if checkpoint is None or 'ep_' not in checkpoint:
                    continue 
                cur_episode_step = int(re.search(r'\d+', checkpoint).group())
                if self.episode_counter < cur_episode_step:
                    self.episode_counter = cur_episode_step
            if self.episode_counter >= 0:
                alice_checkpoints = os.listdir(model_save_path +'/ep_'+str(self.episode_counter) + '/alice')
                self.alice_step = int(re.search(r'\d+', alice_checkpoints[0]).group())
                self.alice.load(model_save_path + '/ep_'+str(self.episode_counter) + '/alice', self.alice_step)

                bob_checkpoints = os.listdir(model_save_path +'/ep_'+str(self.episode_counter) + '/bob')
                self.bob_step = int(re.search(r'\d+', bob_checkpoints[0]).group())
                self.bob.load(model_save_path + '/ep_'+str(self.episode_counter) +'/bob', self.bob_step)
                self.episode_counter += 1
            
            try:
                with open(self.log_dir + '/' + self.args.exp_name + "_evals.pkl", 'rb') as f:
                    self.eval_scores = pickle.load(f)
            except:
                print ("Could not load old eval scores.")
                pass

        if self.episode_counter == -1:
            self.episode_counter = 0

        self.alice_replay_buffer = ReplayBuffer(self.env.observation_space["full_state_with_goal"].shape,
                                                self.env.action_space.shape,
                                                int(args.replay_buffer_capacity),
                                                self.device)

        
        self.bob_replay_buffer = ReplayBuffer(self.env.observation_space["full_state_with_goal"].shape,
                                                self.env.action_space.shape,
                                                int(args.replay_buffer_capacity),
                                                self.device)
        
        if not os.path.exists(model_save_path + '/alice_replay_buffer'):
            os.mkdir(model_save_path + '/alice_replay_buffer')
        if not os.path.exists(model_save_path + '/bob_replay_buffer'):
            os.mkdir(model_save_path + '/bob_replay_buffer')
        
        if reload_model:
            self.alice_replay_buffer.load(model_save_path + '/alice_replay_buffer')
            self.bob_replay_buffer.load(model_save_path + '/bob_replay_buffer')
        
        if args.use_bc:
            self.bob_bc_buffer = BCReplayBuffer(self.env.observation_space["full_state_with_goal"].shape,
                                        self.env.action_space.shape,
                                        int(args.replay_buffer_capacity),
                                        self.device)
            self.alice_bc_buffer = BCReplayBuffer(self.env.observation_space["full_state_with_goal"].shape,
                                        self.env.action_space.shape,
                                        int(args.replay_buffer_capacity),
                                        self.device)
            
            if not os.path.exists(model_save_path + '/alice_bc_buffer'):
                os.mkdir(model_save_path + '/alice_bc_buffer')
            if not os.path.exists(model_save_path + '/bob_bc_buffer'):
                os.mkdir(model_save_path + '/bob_bc_buffer')

            if reload_model:
                self.bob_bc_buffer.load(model_save_path + '/bob_bc_buffer')
                self.alice_bc_buffer.load(model_save_path + '/alice_bc_buffer')
        
        if args.use_bc or args.use_her or args.num_subgoals > 1:
            # Temporary buffers for most recent trajectory, used for relabelling
            self.temp_traj_buffer_alice = ReplayBuffer(self.env.observation_space["full_state"].shape,
                                        self.env.action_space.shape,
                                        int(args.num_steps_alice),
                                        self.device,
                                        goal_shape=self.env.observation_space["xyz_state"].shape)
            self.temp_traj_buffer_bob = ReplayBuffer(self.env.observation_space["full_state"].shape,
                                        self.env.action_space.shape,
                                        int(args.num_steps_bob),
                                        self.device,
                                        goal_shape=self.env.observation_space["xyz_state"].shape)

        # to compute utilities
        self.alice_rewards = []
        self.bob_rewards = []

        debug_video_dir = os.path.join(log_dir, "debug_video")
        utils.cleanup_log_dir(debug_video_dir)
        self.debug_video = VideoRecorder(debug_video_dir if args.save_video else None)
        self.video_recorder = VideoRecorder(debug_video_dir if args.save_video else None)

        self.num_evals = 0
   
    def evaluate(self, eval_env, eval_rng, step,  is_alice=False):
        agent = self.bob
        agent_str = '_bob'
        if is_alice:
            agent = self.alice
            agent_str = '_alice'

        #Evaluate with random_corner
        goal_shape = eval_env.get_attr('goal', indices=0)[0].shape 
        goal_space = eval_env.env_method('goal_space')[0]
        test_cases = get_test_cases(self.args.env_name)
        for test_case in test_cases:
            average_episode_reward = 0
            average_success_rate = 0
            for episode in range(self.args.num_eval_episodes):
                eval_env.env_method('set_init_qpos_qvel')
                eval_env.env_method('sim_update_qpos_qvel')
                success = torch.zeros(self.args.num_processes)
                goals = generate_eval_goals(self.args.num_processes, self.args.env_name, goal_shape, goal_space, eval_rng, test_case)
 
                for i in range(self.args.num_processes):
                    eval_env.env_method('set_goal', 
                                self.args, 
                                goal_space, 
                                goal_shape, 
                                goals[i],
                                indices=i)

                obs = eval_env.reset()
                agent.reset()
                if step % self.args.debug_video_freq == 0:
                    self.video_recorder.init(enabled=(episode==0))

                done = False
                episode_reward = 0
                while not done:
                    with utils.eval_mode(agent):
                        action = agent.act(obs["full_state_with_goal"], sample=False)
                    obs, reward, done, infos = eval_env.step(action)
                    if step % self.args.debug_video_freq == 0:
                        self.video_recorder.record(eval_env)
                    episode_reward += reward
                    for i, info in enumerate(infos):
                        try:
                            success[i] = info["success"] or success[i]
                        except:
                            success[i] = torch.FloatTensor([info["success"]]) or success[i]

                average_episode_reward += episode_reward
                average_success_rate += success.mean()
                if step % self.args.debug_video_freq == 0:
                    self.video_recorder.save(test_case + '_' + str(goals) +'_'+str(step)+'.mp4')

            average_episode_reward = np.sum(average_episode_reward)
            average_episode_reward /= self.args.num_eval_episodes
            average_success = average_success_rate / self.args.num_eval_episodes
            self.logger.log('eval/'+ test_case + agent_str + '/episode_reward', average_episode_reward,
                            step)
            self.logger.log('eval/'+ test_case + agent_str + '/success', average_success,
                            step)
            self.logger.dump(step)
            # TODO: clean this up
            if test_case + agent_str + '_success' in self.eval_scores.keys():
                self.eval_scores[test_case + agent_str + '_success' ].append([average_success, step])
                self.eval_scores[test_case + agent_str + '_rew' ].append([average_episode_reward, step])
            else:
                self.eval_scores[test_case + agent_str + '_success' ] = [[average_success, step]]
                self.eval_scores[test_case + agent_str + '_rew' ] = [[average_episode_reward, step]]

            f = open(self.log_dir + '/' + self.args.exp_name + "_evals.pkl","wb")
            pickle.dump(self.eval_scores,f)
            f.close()


    def relabel_data_bc(self, agent): # where agent == agent to imitate
        if self.args.use_bc:
            if agent == ALICE:
                cur_agent = self.bob 
                bc_buffer = self.bob_bc_buffer
                temp_traj_buffer = self.temp_traj_buffer_alice
            elif agent == BOB:
                cur_agent = self.alice 
                bc_buffer = self.alice_bc_buffer
                temp_traj_buffer = self.temp_traj_buffer_bob
            
            final_obs = torch.as_tensor(temp_traj_buffer.xyz_state[-1]).to(self.device)
            obses, actions, rewards, next_obses, not_dones, not_dones_no_max = temp_traj_buffer.sample(self.args.num_steps_alice, random=False)

            for transition in zip(obses, actions):
                obs, action = transition
                obs = torch.cat((obs, final_obs), dim=-1)

                with torch.no_grad():
                    dist = cur_agent.actor(obs)
                    log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                if torch.isnan(log_prob) or torch.isneginf(log_prob.detach())  or torch.isposinf(log_prob.detach()):
                    continue
                bc_buffer.add(obs.cpu(), action.cpu(), log_prob.cpu())
        return 
    
    def relabel_data_her(self, agent):
        if self.args.use_her:
            if agent == ALICE:
                temp_traj_buffer = self.temp_traj_buffer_alice
            elif agent == BOB:
                temp_traj_buffer = self.temp_traj_buffer_bob

            final_obs = torch.as_tensor(temp_traj_buffer.xyz_state[-1]).to(self.device)
            obses, actions, rewards, next_obses, not_dones, not_dones_no_max = temp_traj_buffer.sample(self.args.num_steps_alice, random=False)

            for step, transition in enumerate(zip(obses, actions, rewards, next_obses, not_dones, not_dones_no_max)):
                if step + self.args.HER_k >= temp_traj_buffer.idx or \
                    step + self.args.HER_k >= temp_traj_buffer.capacity and temp_traj_buffer.full:
                    break

                obs, action, reward, next_obs, not_done, not_done_no_max = transition
                future_xyz, cur_xyz = temp_traj_buffer.sample_future_states(self.args.HER_k, step)

                done = ~not_done.int()
                done_no_max = ~not_done_no_max.int()
                    
                for future_ob in future_xyz: 
                    relabeled_obs = torch.cat((obs, future_ob), dim=-1)
                    relabeled_next_obs = torch.cat((next_obs, future_ob), dim=-1)

                    dist_to_goal = np.linalg.norm(future_ob.cpu() - cur_xyz)
                    if self.args.env_name =='fetch_push' or 'manipulator_reach_multi' == self.args.env_name:
                        reward = 0 if dist_to_goal < .1 else -1
                    else:
                        reward = -dist_to_goal

                    if agent == ALICE:
                        self.alice_replay_buffer.add(relabeled_obs.cpu(), action.cpu(), reward, relabeled_next_obs.cpu(), done, done_no_max)
                    elif agent == BOB:
                        self.bob_replay_buffer.add(relabeled_obs.cpu(), action.cpu(), reward, relabeled_next_obs.cpu(), done, done_no_max)
        return 

    def relabel_data_asp(self, goal, bob_success, bob_reward):
        # relabel most recent trajectory with final state as goal, only for ASP
        temp_traj_buffer = self.temp_traj_buffer_alice

        obses, actions, rewards, next_obses, not_dones, not_dones_no_max = temp_traj_buffer.sample(self.args.num_steps_alice, random=False)
        goal = torch.squeeze(goal)
        for step, transition in enumerate(zip(obses, actions, rewards, next_obses, not_dones, not_dones_no_max)):
            obs, action, reward, next_obs, not_done, not_done_no_max = transition
            relabeled_obs = torch.cat((obs.cpu(), torch.zeros_like(goal)), dim=-1)
            relabeled_next_obs = torch.cat((next_obs.cpu(),  torch.zeros_like(goal)), dim=-1)
            if step == self.args.num_steps_alice - 1:
                if self.args.asp_reward == 'dense':
                    reward = -bob_reward 
                else:
                    reward = 1-bob_success
                done = True
            else:
                reward = 0
                done = False
            done_no_max = False
            self.alice_replay_buffer.add(relabeled_obs, action.cpu(), reward, relabeled_next_obs, done, done_no_max)
        return 

    def compute_agent_utilities(self, gamma, agent, step, log=True):
        utility = torch.zeros(self.args.num_processes).to(self.device)
        if agent == ALICE:
            for n, alice_rew in enumerate(self.alice_rewards):
                utility += alice_rew * torch.FloatTensor([gamma ** n]).to(self.device)
            if log:
                print("Alice Reward: {}".format(torch.sum(torch.FloatTensor(self.alice_rewards), dim=0).mean()))
                self.logger.log('train/alice/reward', torch.sum(torch.FloatTensor(self.alice_rewards), dim=0).mean(), step) #log the modified reward for alice
            self.alice_rewards = []
        else:
            for n, bob_rew in enumerate(self.bob_rewards):
                utility += bob_rew * torch.FloatTensor([gamma ** n]).to(self.device)
            if log:
                print("Bob Reward: {}".format(torch.sum(torch.FloatTensor(self.bob_rewards), dim=0).mean()))
                self.logger.log('train/bob/reward', torch.sum(torch.FloatTensor(self.bob_rewards), dim=0).mean(), step) #log sum of rewards for bob TODO: check if dense/sparse
            self.bob_rewards = []

        return utility


    def compute_utilities(self, gamma, step, log=True):
        alice_utility = self.compute_agent_utilities(gamma, ALICE, step, log)
        bob_utility = self.compute_agent_utilities(gamma, BOB, step, log)

        self.init_obs = None
        return alice_utility.unsqueeze(-1), bob_utility.unsqueeze(-1)
        
        
    def save(self, model_save_path, step):
        os.mkdir(model_save_path+'/ep_'+str(step))
        os.mkdir(model_save_path+'/ep_'+str(step) + '/alice')
        os.mkdir(model_save_path+'/ep_'+str(step) + '/bob')

        self.alice.save(model_save_path+'/ep_'+str(step) + '/alice', self.alice_step)
        self.bob.save(model_save_path+'/ep_'+str(step) + '/bob', self.bob_step)
        self.alice_replay_buffer.save(model_save_path+'/alice_replay_buffer')
        self.bob_replay_buffer.save(model_save_path+'/bob_replay_buffer')

        if self.args.use_bc:
            self.alice_bc_buffer.save(model_save_path+'/alice_bc_buffer')
            self.bob_bc_buffer.save(model_save_path+'/bob_bc_buffer')
        return 

    def generate_traj(self, agent_name, goals=None, update=True):
        env = self.env
        args = self.args
        device = self.device

        num_proc = self.args.num_processes
        success = torch.zeros(num_proc)

        env.env_method('set_agent', agent_name)
        if agent_name == ALICE:
            agent_str = 'alice'
            agent = self.alice
            replay_buffer = self.alice_replay_buffer
            max_episode_steps = args.num_steps_alice
            agent_step = self.alice_step
            if self.args.use_bc or self.args.use_her or self.args.num_subgoals > 1:
                temp_traj_buffer = self.temp_traj_buffer_alice
        elif agent_name == BOB:
            agent_str = 'bob'
            agent = self.bob
            replay_buffer = self.bob_replay_buffer
            max_episode_steps = args.num_steps_bob
            agent_step = self.bob_step
            if self.args.use_bc or self.args.use_her or self.args.num_subgoals > 1:
                temp_traj_buffer = self.temp_traj_buffer_bob
        else:
            raise NotImplementedError
    
        if self.args.use_bc or self.args.use_her or self.args.num_subgoals > 1:
            temp_traj_buffer.clear() # clear previous trajectory at the start of each round

        goal_shape = self.env.get_attr('goal', indices=0)[0].shape
        goal_space = self.env.env_method('goal_space')[0]
        episode_reward = 0
        reward_arr = []
        start_time = time.time()
        for i in range(num_proc):
            env.env_method('set_goal', 
                            args, 
                            goal_space, 
                            goal_shape, 
                            goals[i] if goals is not None else None,
                            indices=i)
            print(agent_str, 'Goal', env.get_attr('goal', indices=i))
        obs = env.reset()
        agent.reset()

        if agent_name == ALICE:
            # first obs for alice and bob should be the same
            self.init_obs = obs["full_state_with_goal"]

        if self.episode_counter % args.debug_video_freq == 0:
            self.debug_video.init(enabled=True)
            self.debug_video.record(env)

        for episode_step in range(max_episode_steps-1):
            if agent_step < self.args.num_seed_steps:
                action = [self.env.action_space.sample()]
            with utils.eval_mode(agent):
                action = agent.act(obs["full_state_with_goal"], sample=True)
            
            if agent_step > self.args.num_seed_steps and (replay_buffer.idx > 0 or replay_buffer.full) and update:
                agent.update(replay_buffer, self.logger, agent_step, agent_str=agent_str)

            next_obs, reward, done, infos = self.env.step(action)
            if self.episode_counter % args.debug_video_freq == 0:
                self.debug_video.record(env)

            for i, info in enumerate(infos):
                success[i] = info["success"] or success[i]

            done = float(done[0])
            done_no_max = 0 if episode_step + 1 == max_episode_steps else done
            episode_reward += np.sum(reward) # reward is an array with VecEnv, hence np.sum
            reward_arr.append(np.sum(reward))
            if update and goals is not None:
                replay_buffer.add(obs["full_state_with_goal"], action, reward, next_obs["full_state_with_goal"], done,
                                    done_no_max)
            if (self.args.use_bc or self.args.use_her or self.args.num_subgoals > 1) and update:
                temp_traj_buffer.add(obs["full_state"], action, reward, next_obs["full_state"], 
                        done, done_no_max, xyz_state=obs["xyz_state"])

            agent_step += 1
            obs = next_obs
            if done:
                break
        if update:
            self.logger.log('train/{}/episode'.format(agent_str), self.episode_counter + 1, agent_step)
            self.logger.log('train/{}/duration'.format(agent_str), time.time() - start_time, agent_step)
            self.logger.log('train/{}/episode_reward'.format(agent_str), np.mean(episode_reward),
                                    agent_step)
            self.logger.dump(agent_step, save=(agent_step > 0))

        if agent_name == ALICE:
            self.alice_rewards = reward_arr
            if update:
                self.alice_step = agent_step
        else:
            self.bob_rewards =  reward_arr 
            if update:
                self.episode_counter += 1
                self.bob_step = agent_step
            
        if self.episode_counter % args.debug_video_freq == 0:
            self.debug_video.save(f'%s_%d.mp4' % (agent_str, agent_step))
        if update and args.use_her:
            self.relabel_data_her(agent_name)

        if goals is None:
            return success, obs 
        return success

            
    def imitate(self, bob_bc_rollouts=None, dual_bc=False):
        if self.args.use_bc:
            if bob_bc_rollouts is not None:
                # For ASP -- Convert from PPO alice rollouts to SAC buffer
                for i in range(self.args.num_steps_alice):
                    obs = bob_bc_rollouts.obs['full_state'][i]
                    goal = bob_bc_rollouts.obs['goal_state'][i]
                    obs = torch.cat((obs, goal), dim=-1)
                    action = bob_bc_rollouts.actions[i]

                    with torch.no_grad():
                        dist = self.bob.actor(obs.to(self.device))
                        log_prob = dist.log_prob(action.to(self.device)).sum(-1, keepdim=True)
                    if torch.isnan(log_prob) or torch.isneginf(log_prob.detach())  or torch.isposinf(log_prob.detach()):
                        continue
                    self.bob_bc_buffer.add(obs.cpu(), action.cpu(), log_prob.cpu())
                if self.bob_bc_buffer.idx > 0:
                    self.bob.bc_update(self.bob_bc_buffer, self.logger, self.bob_step, 'Bob')
                self.bob_bc_buffer.clear()
                
            else:
                if dual_bc:
                    if torch.FloatTensor(self.alice_rewards).mean() > torch.FloatTensor(self.bob_rewards).mean():
                        # Alice did better, so have bob imitate alice
                        self.relabel_data_bc(ALICE)
                        self.bob.bc_update(self.bob_bc_buffer, self.logger, self.bob_step)
                        self.bob_bc_buffer.clear()
                    else:
                        self.relabel_data_bc(BOB)
                        self.alice.bc_update(self.alice_bc_buffer, self.logger, self.alice_step)
                        self.alice_bc_buffer.clear()
                else:
                    self.relabel_data_bc(ALICE)
                    if self.bob_bc_buffer.idx > 0:
                        self.bob.bc_update(self.bob_bc_buffer, self.logger, self.bob_step, 'Bob')
                    self.bob_bc_buffer.clear()

    def generate_initial_goals(self, size=100000):
        # FOR GAN
        goals = []
        steps = 0
        obs = self.env.reset()
        while len(goals) < size:
            steps += 1
            if steps > self.args.num_steps_bob: # don't care about done
                steps = 0
                done = False
                obs = self.env.reset()
                # goals.append(torch.FloatTensor(obs['xyz_state']))
                goals.append(torch.FloatTensor(np.concatenate((obs['xyz_state'], [[.25]] * (2 * np.random.rand(1) - 1 )), axis=-1)))
            else:
                with utils.eval_mode(self.bob):
                    action = self.bob.act(obs["full_state_with_goal"], sample=True)
                obs, _, done, _ = self.env.step(action)
                # goals.append(torch.FloatTensor(obs['xyz_state']))
                goals.append(torch.FloatTensor(np.concatenate((obs['xyz_state'],  [[.25]] * (2 * np.random.rand(1) - 1 )), axis=-1)))
        goals = torch.stack(goals)
        return goals.reshape(size // self.args.num_subgoals, goals.shape[1], goals.shape[-1] * self.args.num_subgoals).to(self.device)

    def get_bob_reward(self):
        # Return most recent Bob reward for dense-ASP
        return np.sum(self.bob_rewards)

    def update_regrets(self):
        # FOR ASP, update alice's buffer with bob's updated performance
        cur_agent = self.bob 
        buffer = self.alice_replay_buffer
        obs, actions, old_rews, next_obs, not_done, not_done_no_max, idxs = buffer.sample(int(self.args.num_stale_updates), idxes=True)
        with torch.no_grad():
            act = cur_agent.act(obs.cpu(), sample=False)
            actor_Q1, actor_Q2 = cur_agent.critic(obs, torch.FloatTensor(act).to(self.device))
            actor_Q = torch.min(actor_Q1, actor_Q2)

        buffer.update_regrets(idxs, -actor_Q.cpu(), self.args.stale_regret_coeff) #replace old regrets with Q_value estimates