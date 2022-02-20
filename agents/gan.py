import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def batch_feed_array(array, batch_size):
    data_size = array.shape[0]
    #assert data_size >= batch_size
    
    if data_size <= batch_size:
        while True:
            yield array
    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array[start:start + batch_size, ...]
            else:
                yield torch.cat([array[start:data_size, ...], array[0: start + batch_size - data_size, ...]], dim=0
                )
            start = (start + batch_size) % data_size


class StateGenerator(object):
    """A base class for state generation."""

    def pretrain_uniform(self):
        """Pretrain the generator distribution to uniform distribution in the limit."""
        raise NotImplementedError

    def pretrain(self, states):
        """Pretrain with state distribution in the states list."""
        raise NotImplementedError

    def sample_states(self, size):
        """Sample states with given size."""
        raise NotImplementedError

    def sample_states_with_noise(self, size):
        """Sample states with noise."""
        raise NotImplementedError

    def train(self, states, labels):
        """Train with respect to given states and labels."""
        raise NotImplementedError

class StateGAN(StateGenerator):
    """A GAN for generating states. It is just a wrapper for clgan.GAN.FCGAN"""

    def __init__(self, evaluater_size, goal_space, device,
                 state_noise_level=.5, *args, **kwargs):
        self.gan = FCGAN(
            generator_output_size=goal_space.shape[0],
            discriminator_output_size=evaluater_size,
            device=device,
            *args,
            **kwargs
        )
        self.device = device
        self.state_size = goal_space.shape[0]
        self.evaluater_size = evaluater_size
        self.state_noise_level = state_noise_level
        self.goal_space = goal_space

    def pretrain_uniform(self, size=10000, report=None, *args, **kwargs):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        states = np.random.uniform(
            self.goal_space.low, self.goal_space.high, size=(size, self.state_size)
        )
        return self.pretrain(states, *args, **kwargs)

    def pretrain(self, states, logger, outer_iters=500, generator_iters=1, discriminator_iters=1):
        """
        Pretrain the state GAN to match the distribution of given states.
        :param states: the state distribution to match
        :param outer_iters: of the GAN
        """
        labels = torch.ones((states.shape[0], self.evaluater_size))  # all state same label --> uniform
        return self.train(
            states, labels,logger, outer_iters,  generator_iters, discriminator_iters
        )

    def _add_noise_to_states(self, states):
        clipped_states = [] #states.clone()
        for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
            noise = np.random.randn(*states[..., i].shape) * self.state_noise_level
            clipped_states.append(torch.clamp(states[..., i] + torch.FloatTensor(noise).to(self.device), low, high))
        clipped_states = torch.stack(clipped_states).permute(1,0)
        return clipped_states

    def sample_states(self, size):  # un-normalizes the states
        normalized_states, noise = self.gan.sample_generator(size)
        scaled_goals =  []#normalized_states.clone()
        for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
            space_range = (high - low)/2
            space_mid = (high + low)/2
            scaled_goals.append(normalized_states[..., i] * space_range + space_mid)
            # scaled_goals[..., i] = scaled_goals[..., i] * space_range + space_mid
        scaled_goals = torch.stack(scaled_goals).transpose(1,0)
        return scaled_goals, noise

    def sample_states_with_noise(self, size):

        states, noise = self.sample_states(size)
        states = self._add_noise_to_states(states)
        return states, noise

    def train(self, states, labels,logger, outer_iters,  generator_iters=1, discriminator_iters=1):
        normalized_states = [] #states.clone()
        for i, (low, high) in enumerate(zip(self.goal_space.low, self.goal_space.high)):
            space_range = (high - low)/2
            space_mid = (high + low)/2
            normalized_states.append((states[..., i] - space_mid) / space_range)
        normalized_states = torch.stack(normalized_states).view(*states.shape)

        return self.gan.train(
            normalized_states, labels,logger,  outer_iters,  generator_iters, discriminator_iters
        )

    def discriminator_predict(self, states):
        return self.gan.discriminator_predict(states)


class FCGAN(object):
    def __init__(self, generator_output_size, discriminator_output_size, device,
                 generator_layers=(256, 256), discriminator_layers=(128, 128), noise_size=4):

        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.noise_size = noise_size
        self.device = device

        self.generator = Generator(
                generator_output_size, hidden_layers=generator_layers, noise_size=noise_size, device=device,
            )

        self.discriminator = Discriminator(
                generator_output_size, generator_output_size,  discriminator_output_size, device=device,
                hidden_layers=discriminator_layers, 
            )

        self.generator_optimizer = optim.RMSprop(self.generator.net.parameters(), lr=0.001)
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.net.parameters(), lr=0.001)

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size, batch_size=64):
        generator_samples = []
        generator_noise = []
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.sample_random_noise(sample_size)
            noise = torch.FloatTensor(noise).to(self.device)
            generator_noise.append(noise)
            generator_samples.append(self.generator.forward(noise).clone())
        return torch.vstack(generator_samples), torch.vstack(generator_noise)

    def train(self, X, Y, logger, outer_iters, generator_iters=1, discriminator_iters=1):
        sample_size = X.shape[0]
        train_size = sample_size
        
        batch_size = sample_size
        
        generated_Y = torch.zeros((batch_size, self.discriminator_output_size)).to(self.device)
        
        # batch_feed_X = batch_feed_array(X, batch_size)
        # batch_feed_Y = batch_feed_array(Y, batch_size)
        
        
        for i in range(outer_iters):
            for j in range(discriminator_iters):
                sample_X = X.clone()#next(batch_feed_X).clone()
                sample_X = sample_X.squeeze()
                sample_Y = Y.clone()#next(batch_feed_Y).clone()

                sample_Y = torch.FloatTensor(sample_Y).to(self.device)
                generated_X, random_noise = self.sample_generator(batch_size)
                generated_X = generated_X.squeeze().clone()
                
                train_X = torch.vstack([sample_X, generated_X])
                train_Y = torch.vstack([sample_Y, generated_Y.clone()])
                
                dis_log_loss = self.train_discriminator(
                    train_X, train_Y, 1, no_batch=True
                )
            
            for k in range(generator_iters):
                gen_log_loss = self.train_generator(random_noise, 1)
                if k > 5:
                    random_noise = self.sample_random_noise(batch_size)
                
            logger.log('train/gan/gen_loss', gen_log_loss, i)
            logger.log('train/gan/disc_loss', dis_log_loss, i)

            print('Iter: {}, generator loss: {}, discriminator loss: {}'.format(i, gen_log_loss, dis_log_loss))
                            
        return dis_log_loss, gen_log_loss

    def train_discriminator(self, X, Y, iters, no_batch=True):
        """
        :param X: goal that we know lables of
        :param Y: labels of those goals
        :param iters: of the discriminator trainig
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        if no_batch:
            assert X.shape[0] == Y.shape[0]
            batch_size = X.shape[0]
        else:
            batch_size = 64
        
        # batch_feed_X = batch_feed_array(X, batch_size)
        # batch_feed_Y = batch_feed_array(Y, batch_size)
        
        for i in range(iters):
            train_X = X.clone() #next(batch_feed_X).clone()
            train_Y = Y.clone() #next(batch_feed_Y).clone()
            output = self.discriminator.forward(train_X)
        
            self.discriminator_optimizer.zero_grad()
            loss = self.discriminator.discriminator_loss(train_Y, output)
            loss.backward()
            self.discriminator_optimizer.step()
                
        return loss

    def train_generator(self, X, iters):
        """
        :param X: These are the latent variables that were used to generate??
        :param iters:
        :return:
        """
        log_loss = []
        batch_size = X.shape[0]#64
        
        # batch_feed_X = batch_feed_array(X, batch_size)
        
        for i in range(iters):
            train_X = X.clone()# next(batch_feed_X).clone()
            output = self.discriminator.forward(self.generator.forward(train_X))
        
            self.generator_optimizer.zero_grad()
            loss = self.discriminator.generator_loss(output)
            loss.backward()
            self.generator_optimizer.step() 
               
        return loss

    def discriminator_predict(self, X):
        batch_size = 64
        output = []
        for i in range(0, X.shape[0], batch_size):
            sample_size = min(batch_size, X.shape[0] - i)

            samples = torch.FloatTensor(X[i:i + sample_size]).to(self.device)
            output.append(self.discriminator.forward(samples))
        return torch.vstack(output)


class Generator(object):
    def __init__(self, output_size, device, hidden_layers=(256, 256), noise_size=4):

        self.net = nn.Sequential(*[
                    nn.Linear(noise_size, hidden_layers[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[0], hidden_layers[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[1], output_size),
                    nn.Tanh()
                ]).to(device)

        self.net.apply(init_weights)
    
    def forward(self, x):
        return self.net(x)

class Discriminator(object):
    def __init__(self, generator_output, input_size, output_size, device, hidden_layers=(128, 128)):

        self.net = nn.Sequential(*[
                    nn.Linear(input_size, hidden_layers[0]),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_layers[0], hidden_layers[1]),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_layers[1], output_size),
                ]).to(device)
        
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)    

    def discriminator_loss(self, label, sample_output):
        return torch.mean(torch.square(2 * label - 1 - sample_output))
    
    def generator_loss(self, generator_output):
        return torch.mean(torch.square(generator_output - 1)) # why subtract 1 here?

        