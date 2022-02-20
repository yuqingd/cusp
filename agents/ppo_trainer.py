import torch
import os
from agents.model import Policy
import re
from agents.ppo import PPO
from utils.utils import *
from envs.SelfPlay import ALICE, BOB, get_test_cases, generate_eval_goals
from utils.utils import VideoRecorder
import time
import pickle

class PPOTrainer:
    def __init__(self, args, L, envs, device, log_dir, model_save_path, reload_model=False):
        self.args = args
        self.L = L
        self.env = envs
        self.device = device
        self.episode_counter = 0
        
        self.eval_scores = {}
        self.log_dir = log_dir

        self.alice_step = 0
        self.bob_step = 0
        self.step = 0

        if reload_model:
            for checkpoint in os.listdir(model_save_path):
                try:
                    cur_step = int(re.search(r'\d+', checkpoint).group())
                except:
                    cur_step = 0
                if self.step < cur_step:
                    self.step = cur_step

            if self.step > 0:
                actor_critic_alice = torch.load(os.path.join(model_save_path, "alice_" + str(self.step) + ".pt"))
                actor_critic_bob = torch.load(os.path.join(model_save_path, "bob_" + str(self.step) + ".pt"))
            
            try:
                with open(self.log_dir + '/' + self.args.exp_name + "_evals.pkl", 'rb') as f:
                    self.eval_scores = pickle.load(f)
            except:
                print("Could not load old eval scores.")
                pass
        
        if self.step == 0:
            actor_critic_alice = Policy(goal_conditioned=args.goal_algo != 'asp',obs_space=envs.observation_space, action_space=envs.action_space, device=device).to(device)
            actor_critic_bob = Policy(obs_space=envs.observation_space, action_space=envs.action_space, device=device).to(device)        

        self.alice = PPO(actor_critic_alice, **args.ppo_agent.params)
        self.bob = PPO(actor_critic_bob, **args.ppo_agent.params)
        
        self.alice_rollouts = RolloutStorage(args.num_steps_alice, envs.observation_space, envs.action_space,
                                    self.alice.actor_critic.recurrent_hidden_state_size, num_processes=args.num_processes)

        self.bob_rollouts = RolloutStorage(args.num_steps_bob, envs.observation_space, envs.action_space,
                                            self.bob.actor_critic.recurrent_hidden_state_size,  num_processes=args.num_processes)

        if self.args.use_bc:
            self.bob_bc_rollouts = RolloutStorage(args.num_steps_alice, envs.observation_space, envs.action_space,
                                        self.bob.actor_critic.recurrent_hidden_state_size,  num_processes=args.num_processes)
        else:
            self.bob_bc_rollouts = None
        
        debug_video_dir = os.path.join(log_dir, "debug_video")
        cleanup_log_dir(debug_video_dir)
        self.debug_video = VideoRecorder(debug_video_dir if args.save_video else None)
        self.video_recorder = VideoRecorder(debug_video_dir if args.save_video else None)

    def evaluate(self, eval_env, eval_rng, step,  is_alice=False):
        agent = self.bob
        agent_name = BOB
        agent_str = '_bob'
        if is_alice:
            agent = self.alice
            agent_name = ALICE
            agent_str = '_alice'

        #Evaluate with random_corner, TODO: move test cases to env
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
                self.video_recorder.init(enabled=(episode == 0))
                done = False
                episode_reward = 0
                for episode_step in range(self.args.num_steps_bob):
                    action = self.act(agent_name, episode_step, obs)
                    obs, reward, done, infos = eval_env.step(action)

                    self.video_recorder.record(eval_env)
                    episode_reward += reward

                    for i, info in enumerate(infos):
                        success[i] = info["success"] or success[i]
                    
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                    [[0.0] if 'TimeLimit.truncated' in info.keys() else [1.0]
                    for info in infos])

                average_episode_reward += episode_reward
                average_success_rate += success.mean()
                self.video_recorder.save(test_case + '_' + str(goals) +'_'+str(step)+'.mp4')

            average_episode_reward = np.sum(average_episode_reward)
            average_episode_reward /= self.args.num_eval_episodes
            average_success = average_success_rate / self.args.num_eval_episodes
            self.L.log('eval/'+ test_case +  agent_str + '/episode_reward', average_episode_reward,
                            step)
            self.L.log('eval/'+ test_case +  agent_str + '/success', average_success,
                            step)
            self.L.dump(step)

            #TODO: cleanup
            if test_case + agent_str + '_success' in self.eval_scores.keys():
                self.eval_scores[test_case + agent_str + '_success' ].append([average_success, step])
                self.eval_scores[test_case + agent_str + '_rew'].append([average_episode_reward, step])
            else:
                self.eval_scores[test_case + agent_str + '_success'] = [[average_success, step]]
                self.eval_scores[test_case + agent_str + '_rew'] = [[average_episode_reward, step]]

            f = open(self.log_dir + '/' + self.args.exp_name + "_evals.pkl","wb")
            pickle.dump(self.eval_scores,f)
            f.close()


    def to(self, device):
        self.alice_rollouts.to(device)
        self.bob_rollouts.to(device)
        if self.bob_bc_rollouts is not None:
            self.bob_bc_rollouts.to(device)      
    
    def act(self, agent, env_step, obs, save=True):
        obs = to_tensor(obs, self.device)
        if agent == ALICE:
            hiddens = (self.alice_rollouts.recurrent_hidden_states_ac[env_step], self.alice_rollouts.recurrent_hidden_states_v[env_step])
            with torch.no_grad():        
                value, action, action_log_prob, recurrent_hidden_states = self.alice.actor_critic.act(obs, hiddens, self.alice_rollouts.masks[env_step])

            if save:
                self.alice_rollouts.insert_act(action, action_log_prob, value, recurrent_hidden_states)
        elif agent == BOB:
            hiddens = (self.bob_rollouts.recurrent_hidden_states_ac[env_step], self.bob_rollouts.recurrent_hidden_states_v[env_step])
            with torch.no_grad():        
                value, action, action_log_prob, recurrent_hidden_states = self.bob.actor_critic.act(obs, hiddens, self.bob_rollouts.masks[env_step])

            if save:    
                self.bob_rollouts.insert_act(action, action_log_prob, value, recurrent_hidden_states)
        else:
            raise NotImplementedError

        return action
        
    def after_act(self, agent, next_obs, reward, masks, bad_masks, obs=None, action=None):
        next_obs = to_tensor(next_obs, self.device)
        if agent == ALICE:
            self.alice_rollouts.insert_after_act(next_obs, reward, masks, bad_masks)
        elif agent == BOB:
            self.bob_rollouts.insert_after_act(next_obs, reward, masks, bad_masks)
        else:
            raise NotImplementedError


    def update(self, step):
        print("{} - Alice Reward: {}".format(step, torch.sum(self.alice_rollouts.rewards, dim=0).mean()))
        self.L.log('train/alice/reward', torch.sum(self.alice_rollouts.rewards, dim=0).mean(), step) #log the modified reward for alice

        with torch.no_grad():
            next_value_alice = self.alice.actor_critic.get_value(
                self.alice_rollouts.get_last_obs(), self.alice_rollouts.get_last_recurrent(),
                self.alice_rollouts.masks[-1]).detach()
        self.alice_rollouts.compute_returns(next_value_alice, self.args.use_gae, self.args.gamma,
                                    self.args.gae_lambda, self.args.use_proper_time_limits)
        self.alice_rollouts.to(self.device)

        with torch.no_grad():
            next_value_bob = self.bob.actor_critic.get_value(
                self.bob_rollouts.get_last_obs(), self.bob_rollouts.get_last_recurrent(),
                self.bob_rollouts.masks[-1]).detach()
        self.bob_rollouts.compute_returns(next_value_bob, self.args.use_gae, self.args.gamma,
                                    self.args.gae_lambda, self.args.use_proper_time_limits)
        self.bob_rollouts.to(self.device)

        #PPO update
        alice_value_loss, alice_action_loss, _,  alice_dist_entropy, alice_goal_diversity, alice_goal_magnitude = self.alice.update(self.alice_rollouts)
        self.alice_rollouts.after_update()
        self.L.log('train/alice/policy/value_loss', alice_value_loss, step)
        self.L.log('train/alice/policy/action_loss', alice_action_loss, step)
        self.L.log('train/alice/policy/entropy', alice_dist_entropy, step)
        self.L.log('train/alice/goals/diversity', alice_goal_diversity, step)
        self.L.log('train/alice/goals/magnitude', alice_goal_magnitude, step)

        bob_value_loss, bob_action_loss, bc_loss, bob_dist_entropy, bob_goal_diversity, bob_goal_magnitude = self.bob.update(self.bob_rollouts, bc_rollouts=self.bob_bc_rollouts)
        self.bob_rollouts.after_update()
        self.L.log('train/bob/policy/value_loss', bob_value_loss, step)
        self.L.log('train/bob/policy/action_loss', bob_action_loss, step)
        self.L.log('train/bob/policy/bc_loss', bc_loss, step)
        self.L.log('train/bob/policy/entropy', bob_dist_entropy, step)
        self.L.log('train/bob/goals/diversity', bob_goal_diversity, step)
        self.L.log('train/bob/goals/magnitude', bob_goal_magnitude, step)

    def save(self, model_save_path, step):
        torch.save(self.bob.actor_critic, os.path.join(model_save_path, "bob_" + str(step) + ".pt"))
        torch.save(self.alice.actor_critic, os.path.join(model_save_path, "alice_" + str(step) + ".pt"))

    def compute_utilities(self, gamma, step):
        alice_utility = self.alice_rollouts.get_utility(gamma)
        bob_utility = self.bob_rollouts.get_utility(gamma)

        return alice_utility, bob_utility, None

    def get_bob_reward(self):
        # Return most recent Bob reward for dense-ASP
        # first mask out Bob rewards based on dones
        self.bob_rollouts.rewards = self.bob_rollouts.rewards * self.bob_rollouts.masks[:-1]
        return torch.sum(self.bob_rollouts.rewards)


    def relabel_data(self, goal, bob_success, bob_reward):
        self.alice_rollouts.rewards = torch.zeros_like(self.alice_rollouts.rewards)
        self.alice_rollouts.masks = torch.ones_like(self.alice_rollouts.masks)
        if not torch.is_tensor(bob_reward):
            bob_reward = torch.Tensor([bob_reward])
        if self.args.asp_reward == 'dense':
            self.alice_rollouts.rewards[-1] = -bob_reward # 1-bob_success
        else:
            self.alice_rollouts.rewards[-1] = 1-bob_success

        # Relabel data for BC -- relabel Alice's last rollout with Alice's final state as the goal and use this as Bob's BC buffer
        if self.bob_bc_rollouts is not None and bob_success == 0: # only do BC if Bob failed 
            for env_step in range(self.args.num_steps_bob):
                with torch.no_grad():
                    bob_obs = {k: v[env_step] for k, v in self.alice_rollouts.obs.items()}
                    alice_action = self.alice_rollouts.actions[env_step]
                    bob_obs['goal_state'] = goal

                    hiddens = (self.alice_rollouts.recurrent_hidden_states_ac[env_step],self.alice_rollouts.recurrent_hidden_states_v[env_step])
                    bob_obs = to_tensor(bob_obs, self.device)
                    bc_value, bc_action_log_probs, bc_dist_entropy, bc_rnn_hxs = self.bob.actor_critic.evaluate_actions(
                                        bob_obs, hiddens, alice_action, self.alice_rollouts.masks[env_step])
                    masks = torch.FloatTensor(
                                    [[1.0] if mask else [0.0] for mask in self.alice_rollouts.masks[env_step]])
                    bad_masks = torch.ones_like(masks)
                    reward = torch.FloatTensor([[0]])

                    self.bob_bc_rollouts.insert_act(alice_action, bc_action_log_probs, bc_value, bc_rnn_hxs)
                    self.bob_bc_rollouts.insert_after_act(bob_obs, reward, masks, bad_masks)


    def generate_traj(self, agent_name, goals=None):
        env = self.env
        args = self.args
        device = self.device

        num_proc = self.args.num_processes
        success = torch.zeros(num_proc)

        env.env_method('set_agent', agent_name)
        if agent_name == ALICE:
            agent_str = 'Alice'
            max_episode_steps = args.num_steps_alice
            agent_step = self.alice_step
        elif agent_name == BOB:
            agent_str = 'Bob'
            max_episode_steps = args.num_steps_bob
            agent_step = self.bob_step
        else:
            raise NotImplementedError

        goal_shape = self.env.get_attr('goal', indices=0)[0].shape
        goal_space = self.env.env_method('goal_space')[0]

        episode_reward = 0
        reward_arr = []
        start_time = time.time()
        # num_evals = 0

        for i in range(num_proc):
            env.env_method('set_goal', 
                            args, 
                            goal_space, 
                            goal_shape, 
                            goals[i] if goals is not None else None,
                            indices=i)
            print(agent_str, 'Goal', env.get_attr('goal', indices=i))
        
        obs = self.env.reset()

        if self.episode_counter % args.debug_video_freq == 0:
            self.debug_video.init(enabled=True)
            self.debug_video.record(env)

        for episode_step in range(max_episode_steps-1):
            action = self.act(agent_name, episode_step, obs)
            next_obs, reward, done, infos = self.env.step(action)

            if self.episode_counter % args.debug_video_freq == 0:
                self.debug_video.record(env)

            for i, info in enumerate(infos):
                success[i] = info["success"] or success[i]
            
            masks = torch.FloatTensor([[0.0] if (done_ or success[i] or episode_step == max_episode_steps-1) else [1.0] for i, done_ in enumerate(done)])

            bad_masks = torch.FloatTensor(
            [[0.0] if 'TimeLimit.truncated' in info.keys() else [1.0]
             for info in infos])

            episode_reward += np.sum(reward)
            reward = torch.FloatTensor([reward])
            masks = torch.FloatTensor(masks)
            bad_masks = torch.FloatTensor(bad_masks)

            self.after_act(agent_name, next_obs, reward, masks, bad_masks, obs, action)
            obs = next_obs

        self.L.log('train/{}/episode'.format(agent_str), self.episode_counter + 1, agent_step)
        self.L.log('train/{}/duration'.format(agent_str), time.time() - start_time, agent_step)
        self.L.log('train/{}/episode_reward'.format(agent_str), np.mean(episode_reward),
                                agent_step)
        self.L.dump(agent_step, save=(agent_step > 0))

        if self.episode_counter % args.debug_video_freq == 0:
            self.debug_video.save(f'%s_%d.mp4' % (agent_str, self.step))

        if goals is None:
            return success, obs 
        else:
            return success