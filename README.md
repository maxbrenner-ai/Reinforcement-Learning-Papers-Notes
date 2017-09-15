# Reinforcement Learning Papers Notes
Notes on many interesting RL papers. Also some of the notes are a little messy because these were my personal notes to start off with but most of them should be a pretty understandable summary.

### Important:
Most of the papers on this list were found through [Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf) which is a great source by Yuxi Li. The grouping also follows this source quite heavily.

## Preliminary Papers to Read
I don't include notes for these papers or topics because there are plenty of resources around the web to help understand them, including in [Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf) 
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) DQN
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) Double DQN
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) Dueling DQN
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) A3C
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) DDPG

## Notes:

## Policy Gradient
### Trust Region Policy Optimization (TRPO) (Under construction)
https://arxiv.org/pdf/1502.05477.pdf
- MATH WARNING
- This alg ensures monotonic improvement of policies (aka helps with policy optimization)
- Lots of math involved in the proofs and equations so need to review this some more

### Combining Policy Gradient and Q-Learning (PGQ)
https://arxiv.org/pdf/1611.01626.pdf
- This alg’s goal is to combine Actor-Critic and Q-Learning
- The alg works like normal A2C or A3C for the most part (exactly the same)
- But every transition is saved in a collective memory (if there are multiple agents aka A3C)
- Then we perform Q-Learning using batches from this memory
- The Q-Learning formula works the same as normal Q-Learning
- Formula: Gradient w.r.t param * Q-Leaning Error * (log policy(s,a) + V(s))
    - So you train the actor-critic network with this as the obj function value in addition to the normal
    online A3C updates


## Imitation, Demonstration and Semi-Supervised Learning
### Generative Adversarial Nets (GAN)
https://arxiv.org/pdf/1406.2661.pdf
- The purpose of this alg is to train a generative model (aka a model that can make data aka learn the same
data distribution of the training data)
- It does this by also training its adversary, a discriminator model, that tries to predict if the data it is
seeing came from the generator or the training data
- It boils down to a minimax game between two players so here is what we train each net on:
- Generative NN loss func (aka minimize this func): log(1-D(G(z))) where z is noisy input that the generative
model uses to create data (also the higher D(x) the more it predicts the data is from the training data)
    - An alternative is for it to maximize log(D(G(z)) which helps early on give it stronger gradients
- Discriminator obj func: log(D(x)) + log (1-D(G(z)))
- Its pretty simple but effective
- An additional feature is for there to be more training passes of the discriminator than the generator as
to keep the generator inline, the generators only clue as to how good it is doing is through the discriminator
which should always be kept close to optimal for this reason

### Generative Adversarial Imitation Learning (GAIL)
https://arxiv.org/pdf/1701.07274.pdf
- MATH WARNING
- IRL (Inverse Reinforcement Learning) attempts to learn a reward function based off of an example’s policy
trajectories, however the RL alg part runs in an inner loop of this aka it's a very computationally inefficient
alg, also since in the end we are just trying to make optimal moves with our learning policy we should just
learn the expert policy itself, aka learn how to make actions directly instead of just a reward function,
this is what GAIL does
- Just like in GAN we have a discriminator that tries to distinguish between the expert policy and the imitator
policy
- We gather a set of the experts trajectories and then a set of the imitators trajectories and then every step
of their trajectories we update the discriminator using a supervised kind of method on it. and we update
the imitator using the TRPO rule with log(D(s,a)) as the cost (-log(D(s,a)) as reward)

### Learning from Demonstrations for Real World Reinforcement Learning (DQfD)
https://arxiv.org/pdf/1704.03732.pdf
- Deep Q-Learning from Demonstrations (DQfD)
- This is an alg for imitation learning when you have demonstration data by some expert policy but not the
expert policy itself (unlike in IRL and GAIL which does have an expert policy to generate trajectories)
- Pre-train on demonstration data aka expert policy/agent trajectories
- You have 4 losses: Supervised, Q-Learning 1 step, Q-Learning n-step and L2 Reg used in the pre-training
on the expert trajectories/data
- Minibatch training is done both in the pre-training and normal training but the losses from above
are also used in normal training (the whole alg is DQN which is how you pick the actions, and this is why exp
replay is used (it's just the loss function (the 4 losses added) that is the substitute for normal MSE)

### Generalizing Skills with Semi-Supervised Reiforcement Learning (S3G)
https://arxiv.org/pdf/1612.00429.pdf
- Semi-Supervised Skill Generalization (S3G)
- This alg is used when you have limited labeled data and unlabeled data, in other words you have MDPs or trajectories
with reward labels and also unlabeled experience available from MDPs with no reward labels (this is most of
the algorithm)
- This is called Semi-Supervised RL, where the agent knows the reward function in some settings but not others
- Allows for more realistic use
- The algorithm:
    - Optimize a policy on the labeled experience and then run this policy on the labeled MDPs again and
    gather samples for use in the main algorithm (aka the reason for this part is just for the samples)
    - Then in a loop:
    - run the current policy you have (not the one from above a new one) on the unlabeled MDPs to gather up
    samples, then add these samples to the list you have of the samples from the labeled MDPs and the ones
    you have been collecting as you go from this step
    - Update the reward function approximation we have currently (has its own params) using this list of
    samples
    - Update the current policy again with the samples and current reward func approx
- There are two eq for updating the reward func and policy respectively (kinda weird so didn't write them down)
- So the point of this alg is to gather up samples from labeled MDPs as you normally would (thru optimizing a
policy on them) and then using this along with the samples you gather as you go to update the reward function
approx and policy approx on the unlabeled MDPs which is the majority of the alg
    - IRL
- Note: The reason for learning a reward func instead of just a policy is because decoupling the reward func learning
from the MDP allows us to generalize more readily because it's not attached to a specific MDP


## Planning
### Value Iteration Networks (VIN)
https://arxiv.org/pdf/1602.02867.pdf
- This allows for planning using value iteration
- Not model-based because we aren't directly learning a model of M but planning off of M_
- So if M is the MDP we can find M_ which is an MDP that gives info about the optimal plan for the specific M,
and we have a reward function and transition function associated with M_ as well, where the reward func maps
a state to a reward if near a goal ect. and the M_ trans func that encodes deterministic movements INDEPENDENT
of the current state observation
    - This is very important because it allows for generalization, so if you can train an agent to learn
    a new maze without refreshing it
- The next step is to do VI using M_ to obtain V_*, the VI module is differentiable which allows for end-to-end
training of this network through back prop
- the VI module has the structure of a CNN, feed in the reward func for M_ and the prev V_ and get out all the
Q-values and then feed again and get out V_ and restart until V_* is achieved in this VI loop
- So V_* is the whole func and we don't really need all this info to make one step from our current state
so we use an attention module that combines the current observation and V_* to get all that is needed to make
a move (attention modules effectively reduce the num of params to train)
- There are many downfalls to this alg being used in realistic settings, such that you need to be able to
obtain the state space or some kind of rep of it like a graph if its too large, and the reward and transition functions
for M_ are crafted specific to the problem at hand
- Overall it shows how to put planning in a network for end-to-end training and also shows how generalizations
can be made through planning


## Exploration
### Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models
https://arxiv.org/pdf/1507.00814.pdf
- Exploration bonuses can help with exploring, called "optimism under uncertainty"
    - r = r + beta * N(s,a)
    - the reward has a novelty function value added to it
    - By adding to the reward we can train an alg like a DQN to take this route more often (at least until
    the novelty bonus goes down)
- So we have a MLP that acts as a model of the dynamics of the system, where we send in an encoded current state
and the action taken and output the predicted next state (encoded version)
    - So we use an encoded version because we can't send in all of the pixels and expect an accurate construction
    of the next state at all (all the pixels again)
    - So we use a stacked autoencoder to encode the states
- So we compare the encoding of the next state we got in the transition with what the model (MLP) says the
next state was supposed to be (send in the encoded current state AND the action taken) and then compare
the difference by using an euclidean loss function, the novelty bonus is then just the normalized error
    - This works because the more we see a s-a pair the better our model is at predicting the next state
    so if our error is large than that means we haven't seen that s-a pair very much and therefore
    we need to explore it more (larger novelty bonus)
        - Of course we won't see an exact s-a pair again like at all but approx has the same effect
- So we train the model every some odd iterations based on exp replay
- And we train the autoencoder even less (because it takes a long time to train since it must reconstruction and compare
the WHOLE state aka all the pixels every time)

### Deep Exploration via Bootstrapped DQN (Under Construction)
https://arxiv.org/pdf/1602.04621.pdf
- MATH WARNING
- Don't really understand this one but apparently it works better than
"Incentivizing Exploration In Reinforcement Learning With Deep Predictive Models" for exploration

### Hindsight Experience Replay
https://arxiv.org/pdf/1707.01495.pdf
- This alg helps deal with sparse reward environments by using goals to add feedback (UVFA)
- This alg can train an agent on completing multiple goals or just one, it works better with sparse and binary
feedback (0 for not completing a goal and 1 for completing the goal)
- There are two kinds of goals, the main one(s) which have sparse feedback and then goals we make
after an episode is completed to add more feedback (replay goals)
- All goals are either a state itself or a property of a state, this way we can generate a goal when we need to
- So the main part of this alg is we run the episode in its entirety (no training, save the transitions in
short term memory at first) and then we generate a small sample of goals based off of the final (termination)
state of the episode, we then go back through the episode we saved and then save transitions in the main memory
for replay
- So we replay the episode with different goals (generated by some strategy from the termination state) and
with the original goal we had (the true goal of the agent) at each step of the episode replay (we need to do this
because our added goals are based off of the termination state and therefore we need to run through the whole thing
first)
- What this does is it trains the agent's value function on the true goal(s) which are sparse and adds feedback
based off the termination state replay goals
- So as long as the strategy for generating replay goals is somewhat good (maybe indicative of the real goals)
than they are useful in at least guiding the agent to take actions that aren't random but have some sort of
pattern and that coupled with the true goals affecting the value func (no matter how sparse in reward) will allow
the agent to get the feedback it needs to slowly learn how to achieve the real goals down the line


## Attention
### Recurrent Models of Visual Attention (RAM)
https://arxiv.org/pdf/1406.6247.pdf
- Recurrent Attention Model (RAM)
- This is a alg/model that is used to process images (classification or even video game playing can be applied)
- It makes it so the size of the images in the env are independent of the number of params and computation time
needed which is different from a CNN that linearly increases in params with size of images
- aka it's supposed to decrease computation time compared to using a CNN to process the whole image
- Also it helps with cluttered images because it can learn to focus on important parts
- RAM uses an RNN to process an image (RNN because it only sees part of the image at a time therefore it is a POMDP)
- Every step the output of the RNN is which location to process next and some action (action is dependent on the
type of problem, say it was obj detection then it would be a guess at the object class, or if it was playing
a game it would be an action to take in the env)
- The parts of the model are:
    - a sensor which looks at a certain region of the current image and processes whats
    called a glimpse which is just a vector of differing resolutions of the location
    - the main glimpse network takes this sensor's glimpse rep output and the location in the image
    at which this was processed and uses ReLU layers to combine them and output a final feature vec
    of the glimpse
- This is then fed into the RNN unit every step (along with the past hidden rep of the RNN obviously) and the
output is a new location and an action (there are models attached to the output of the RNN that produce the
action and new location)
- So this also gets a new state and a reward from the env at the end of every step
- REINFORCE is used to train the params of the model (all params aka the glimpse network, RNN core and action and
location model params) from the cumulative rewards (or disc rewards)
- The params are then updated based off of maximizing log likelihood of the action using the REINFORCE return
and then doing BPTT with these gradients to update the params
- this had comparable achievement to CNNs

### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
https://arxiv.org/pdf/1502.03044.pdf
- This paper shows how to caption images
- Unlike the paper "Recurrent Models of Visual Attention (RAM)" the attention mechanism used in this is here
not to decrease computation time but to focus on specific features of an image every time step and output a word
- The structure of this is like encoder-decoder
- The encoder is a CNN that process the whole image
    - It creates a list of feature vectors called annotation vectors
    - It picks these vectors from lower levels of the CNN, aka not after any dense layer, so that they
    correspond to different locations on the image (allows for focus on different parts of image)
    - This is not like a seq-seq kind of thing because all the encoder is is a CNN not part of the LSTM
- The decoder is a LSTM that uses the annotation vectors to output a caption
    - There is an attention model that is given the prev hidden state of the LSTM each time step and outputs
    weights for every annotation vector that was made in the encoder and then creates a combo of these
    weighted vectors to create a single context vector which is fed as input each step into the LSTM
    (the encoder's annotation vectors it makes don't change at all, they are just combined in different
    ways in the decoder)
    - Then given this context vector (and the prev hidden state OBVIOUSLY) the LSTM outputs a word
- There are two different ways to do the attention model (although the structure is a MLP either way):
    - Stochastic "Hard" Attention (performed the best in general): this uses REINFORCE to train
    - Deterministic "Soft" Attention
    - Don't understand either really, there is a lot of math involved
- So the encoder CNN learns how to find features in dif locations as it goes and send this to the decoder,
the attention model in the decoder uses the prev hidden state of the LSTM (which represents what's already been
focused on and what's been output) and the annotation vecs (features at dif locations) to create a representation
of what should be focused on, aka given what's already been focused on it can learn how to pick other regions
to focus on to actually understand what it's seeing in the image, and then the LSTM takes the context vector
which is what it should focus on and learns to decode this (aka read it meaningfully) and give this and what its
already output (to keep it grammatical aka natural language) to output another word that describes what it is
seeing overall

### Neural Machine Translation by Jointly Learning to Align and Translate 
https://arxiv.org/pdf/1409.0473.pdf
- This is a new encoder-decoder structure/algorithm for neural machine translation
- encoder-decoder means an encoder rnn reads in a source sentence into a fixed length vector which is then
input into the decoder at each step (along with the prev output word)
- in this case we have an attention mechanism added called an alignment model which allows the decoder at each
time step to look at the hidden states of the encoder at each time step and weight and combine them into
a context vector to use as input to output its best prediction of the next translated word
    - this makes it so the encoder does not need to encode the source sentence all into one fixed length
    vector but allow the decoder to selectively choose (attention) which encoder info to look at
    - this is like the Image Captioning alg because that one also weighted annotation vectors into a context
    vector to use at each time step of an RNN instead of encoding into one fixed-length vector
- So the weights of the annotation vectors (aka the hidden state combo of the two tier bidirectional encoder rnn)
are decided by using an alignment model (feed-forward NN which is trained with the rest of the model) which
represents the attention mechanism
- This aligns model takes as input the prev hidden state of the decoder RNN and the current annotation vector
of the encoder it is judging and scores how well the source words/encoded info in general around the annotation
vectors position matches the output that will occur at that current decoder timestep
- pretty much it represents the probability that the target word that is about to be generated is aligned to aka
translated from the source words around the hidden state/annotation vector (because its bidirectional) and then
if it says it has a high correlation it weights that annotation vector highly so it has more of an effect on
what the decoder will output at that timestep as the translation
- this works because the decoder at each time step not only knows it's prev output but also will find out
what info from the encoder is most likely to help it output the next correct translated word through matching
aka aligning the prev hidden state (decoder's) aka what has been output with the each annotation vec its judging
to say a higher match must mean this annotation vector matters more at this time step for the correct translation
- The encoder is a bidirectional rnn which means it has a layer that reads forward and one that reads backward
(starting at the opposite end) and at each timestep these layers hidden states are combined


## Unsupervised Learning
### Reinforcement Learning with Unsupervised Auxiliary Tasks (UNREAL)
https://arxiv.org/pdf/1611.05397.pdf
- We add aux tasks to help in an env with sparse rewards
- The aux tasks are categorized into control and reward prediction
    - The summary is that learning to better predict certain things and control certain things should
    be rewarded in and of themselves
- Control Tasks: The agent learns to control the env through its actions
    - Pixel Changes (Pixel Control): Reward for maximally changing the intensity of pixels thru actions
    - Feature Control: Reward for maximally activating the network's hidden layers nodes because each node
    represents a specific feature so being able to change them (control them) means you have full control
    and knowledge of all learned features
    - Each of these has its own policy for picking actions given a state to maximize these aux task rewards
- Reward (Prediction) Tasks: Help improve reward prediction and value estimation in general
    - Reward Prediction: Given a few states have it predict what the next imm reward will be
     - Just allows it to better learn what good state sequences look like
    - Value func replay: In addition to normal value approximator improvement in A3C we replay transitions
    from the memory and improve the value func
        - Gives the value approximator a bit of a boost
- The base alg is A3C built with CNN and LSTM networks (LSTM because we need help remembering the past since
each state is partially observed aka not a full summary of the past)
- Each aux function uses its own DQN and model to do its thing and these models are trained to get better at
the aux tasks, but in all cases the params are shared somehow between these aux task models and the main
A3C (LSTM and CNN) model to balance improvement of the aux tasks with the improvement of the main task at hand
- Also these aux tasks use Q-Learning (Off policy) so we can use a memory to train them, the transitions sampled
from this memory are samples using a simplified Prioritized Replay method where there is a 50/50 chance to pick
a transition with a non-zero imm reward and one with a zero reward, this helps with allowing to train on
transitions with feedback more often
- *** This alg is the state-of-the-art rl alg for sparse reward signals, it is better than vanilla A3C, Prioritized Dueling Double DQN ect. according to the authors


## Hierarchical RL
### Strategic Attentive Writer for Learning Macro-Actions (STRAW)
https://arxiv.org/pdf/1606.04695.pdf
- This architecture is called STRategic Attentive Writer (STRAW) and is used to learn macro-actions (aka action
sequences)
- Macro-actions are helpful because they can help with exploration and learning by finding more complex
behavior patterns by chaining primitive actions in a meaningful way
- STRAW learns to follow a plan when useful or replan when useful (a plan means a sequence of actions)
- STRAW is an LSTM with two modules (action plan and commitment plan)
- The action plan is a 2d matrix with time and all actions on the two axis, each value is the probability
of taking that action (softmax) at that timestep (every column is all the actions probabilities)
- the commitment plan is a vector that is on the same axis of time (max time for these is a hyperparameter, it
pretty much represents how long a macro action can be maximally) and each value is a 1 or 0 which represents
if it is following the current action plan at that time step (0) or replacing the action plan AND commitment
plan itself (1), so it will learn to chain 0s together so we can actually get a macro-action to follow and then
add 1s in the replanning of the commitment plan to change the current macro action appropriately
- The two modules are replaced through an attention model (that is differentiable like a NN so it can use BP)
    - the gist is 1 observation can hold enough info to generate a seq of actions so we use an attention
    mechanism to focus on the part of the plan that this observation can really help improve (aka the whole
    action plan is not changed but part of it based off of itself along with the current observation)
    - this attention model is trained to update the action plan and commitment plan more effectively by
    learning to focus on the right parts given a current observation (the whole commitment plan is updated
    however)
- Note: When the action plan is being committed to aka the commitment plan is currently 0 then after each step
each is shifted over aka the first column is removed and the last one has a blank column added to it
- This was updated using a weird loss function but it used A3C

### Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation (h-DQN)
https://arxiv.org/pdf/1604.06057.pdf
- Intrinsic motivation means exploring for exploration's sake, in this paper it means more specifically:
- solving subproblems and goals and chaining them together to allow for more efficient exploration to
solve the whole problem
    - aka the main goal of this is for better exploration to solve the problem
- So this isn't for solving really the credit assignment problem it's for solving a problem where there are
hardly any points of feedback and therefore you need to explore to get to these sparse points
- Alg called h-DQN
- So we have a controller and a meta controller, the controller (DQN) takes an action given a state and goal,
the current goal is given by the meta-controller, once the controller has reached the goal the meta-controller
(also a DQN) outputs a new goal given the current state, and the cycle repeats
- The controller is hooked up to a critic (not in the sense of AC) which just gives it intrinsic rewards as it
makes moves, these transitions are stored in a memory and a batch is selected from the memory at each time step
to train it
- As the controller continues to take actions extrinsic rewards (from the env) are saved up and once the
goal is reached (or the episode terminates) the meta-controller uses the total amount of extrinsic reward
received and saves that transition (aka what goal was chosen from the state and end state and the total rew)
and also gets trained by replays of this memory (replay/training occurs at same time as controller trains)
- The temporal abstraction comes because the meta-controller function of getting a new goal (aka feed forward)
occurs for a fraction of the time the controller is outputting and taking new actions, aka the controller action
taking loop is inside the meta-controller goal producing loop
- Another paper on intrinsic motivation for exploration using a pseudo count based method
https://arxiv.org/pdf/1606.01868.pdf (HARD PAPER, pretty much just math)

### Stochastic Neural Network for Hierarchical Reinforcement Learning
https://arxiv.org/pdf/1704.03012.pdf
- this is an hierarchical architecture (THIS was a hard paper and might wanna read again once i know about SNNs)
- it is for solving the problem of sparse rewards by adding skills (like in the minecraft paper) aka macro
actions
- This is dif from the minecraft paper which had explicit/predefined skills it was trying to train as this one
uses an SNN (Stochastic Neural Network (an RBM is an example of one)) on a pre-training env to learn a
variable number of skills
- So a pre-training env is set up with proxy rewards, this env is simple and the rewards are NOT explicit in what
skills they are trying to train the agent on, but like if it was robot it would get reward for for moving
its arm at high speeds without dropping something in its hand, stuff like that more implicit to skills
    - By using a SNN it can train multiple skills that are predefined i.e it learns what skills are useful
    and how do them each effectively
- Then in the real env the params of the SNN are frozen as is and it is hooked up to a manager neural network
(the higher level of this hierarchical architecture) which can pick a skill from the SNN to use given the
current state until a termination time step (aka just like the minecraft one) and this is trained to use these
more effectively
- the SNN and MNN are trained with TRPO

### A Deep Hierarchical Approach to Lifelong Learning in Minecraft (H-DRLN)
https://arxiv.org/pdf/1604.07255.pdf
- Hierarchical Deep Reinforcement Learning Network (H-DRLN)
- This is a 'lifelong learning' network
- a lifelong learning system is one that leans new skills and retains each of their knowledge as well as
picks which skill to use when it's needed
- A skill is a temporally extended action/option/macro action which just means it's a sequence of actions
instead of just a primitive one
- Deep Skill networks which represent each skill, these are pretrained on their own and also normal trained
with the rest of the architecture, each one is trained using DQN (they are pretrained aka not trained when
the whole H-DRLN is being used, they are trained separately before)
- A deep skill module combines these skill networks
    - There is a DSN array which just holds each DSN separate
    - or a Multi-skill distillation network which combines the layers of each skill and then only the
    output layer is separate for each skill
        - Distillation is the transfer of knowledge from a teacher to a student or in this case
        multiple teachers to a student
- The H-DRLN takes the current and 3 prev states (CNN processes the raw pixels) and the H-DRLN outputs the
Q-values for all primitive actions as well as all the different skills, and DQN is used to pick an action
and does not train the DSNs only the rest of it, this is because the DSNs are pretrained on the tasks
- So really the H-DRLN is trained on how to pick the best primitive action or skill just like a normal DQN
because each DSN is trained separately of it and then there knowledge is distilled together if using the
multi-task distiller module


## Learning to Learn
### Learning to Reinforcement Learn
https://arxiv.org/pdf/1611.05763.pdf
- This allows a low-level system which is an RNN to learn a task while having a high-level system (RL alg)
do slower general learning over multiple different tasks (although from the same family as one another), this
allows the rnn to actually learn each different task faster the more tasks it trains on
- This is to help with data efficiency and generalization
- the RNN is actually learning to implement its own RL procedure for each specific task using the higher-level
RL procedure to guide its param training
- This alg comes from the idea that you need prior biases to learn with less examples, and these biases can
be achieved through having engineers put it into the system (like using a CNN) or through meta-learning
- So once again there is an overarching RL procedure (A3C in this case) that goes through a group of different,
but similar in the underlying principles, tasks while training an RNN which actually implements the lower-level
RL procedure that specifically finds the best policy for each specific task, then at the end of the task the
RNN is refreshed but the overarching RL procedure has params (A3C params) that are trained from all of these
different tasks and generalize between them (through a specific alg which i don't fully understand) so the RNN
can then implement a RL procedure for proceeding tasks that works even faster as it goes
- An important note is that the RL procedure that the RNN implements is usually much different than the
high level one, which shows generalization
- Another note is that once the high-level sys is trained enough you could freeze its training of the RNN
on the specific task and it will still improve

### RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning
https://arxiv.org/pdf/1611.02779.pdf
- Like the paper "Learning to Reinforcement Learn"
- The explanation of at least the RNN was a little clearer, so what is sent in as input at each time step is
the prev action, current state, reward and termination flag from prev action
- The output is an action or distribution of actions, aka you can think of each step of the rnn as the policy
- The hidden state params represent the RL alg
- you have to use an RNN and not like a MLP that resets every step because just like how a normal RL procedure
is through time we need this one to be too so that's why we use an RNN so it can improve upon itself every step
of the way (only refreshes after a full MDP task is done)
    - So another way to think about it is a normal RL alg has the whole history affect its weights so we
    use an RNN so that the whole history can also affect it
    - This is why we only refresh after every task because we don't care about the history but the meta RL
    helps initialize the RNN weights to something useful (generalization)
- This one use TRPO as the RL alg for the outer RL loop instead of A3C like the other one


## Imagination-Based Learning
### Metacontrol for Adaptive Imagination-Based Optimization
https://arxiv.org/pdf/1705.02670.pdf
- This architecture is used for balancing computation time with performance because the one-model-fits-all approach
is not very good as if something is easy you don't want the same complex model working on it as you did a more
difficult problem and visa versa
- A meta controller learns to optimize a seq of imagined internal simulations (aka its model based)
    - So this learns to plan using a model in an imagination sort of way
- There are experts and a controller, the controller produces a control (policy) through the help of experts
that eval the current control
- Every iteration the controller produces a control given the total history of what's happened (past controls,
expert evals ect.), and then an expert evals its quality (eval)
- So we have a performance loss defined like we normally would but also add a resource loss which means how long
it would take under the experts, we are trying to minimize this loss (this is where the trade off is between
performance and time)
- We have a meta controller made of a controller, list of experts, a manager and a memory
- The manager is a meta-level policy that decides whether to terminate the optimization and execute or
to optimize on a different expert, given the history (aka both the controller and manager gets the history)
- There are all MLPs with the exception of the memory which is an LSTM
- They all are trained through minimizing the total loss
- The whole system learns better experts, while learning which expert to pick, and learns when to execute, and
learns how to effectively encode the memory, learn to pick a control policy all to minimize the performance AND
computation time
- This works specifically for contextual bandits

### Learning Model-based Planning from Scratch (IBP)
https://arxiv.org/pdf/1707.06170.pdf
- Imagination-Based Planner (IBP)
- This algorithm/architecture learns to construct, evaluate and execute plans (it is a model-based alg)
- Most model-based alg before could eval a plan but not construct one, and they req problem specific knowledge
and engineering or used tree search like MCTS which requires a perfect game
- An agent can imagine using a model to plan what action it should take at a given timestep
- Aka before making a move it can imagine different scenarios and use this to build a policy which it will
follow when execution time arrives (it can also decide to just execute and not plan at any timestep)
- It is made up of a manager, controller, model and memory
- Every iteration (means every time 1 step occurs in the real env) the manager decides whether to execute without
planning or plan first, the controller is the state to action policy that decides what action to take given a
state (and history), it does this even if imagining, then the model evals the action aka it just predicts the
next state and reward and this all is aggregated to memory aka fed into the memory which is an LSTM and this encodes
a vector rep of the history
- Specifically the manager is a MLP that represents a policy, input: memory encoding, output:
a route which means act, or pick any state in which to imagine a next action from (aka picks imagine)
    - There are three hyperparameter ways to let a manager pick an imagined state in the current iteration
    (once again this means act was not selected) from which to pick an action from to form a plan
        - 1 step: means it always has to pick the first state aka the current actual state it's in and
        this creates a depth-1 tree
        - n step: means it always has to pick the last imagined state that was led to from the controller
        and model predictions, creates a n-depth chain
        - tree: means it can pick any prev imagined state (most flexible option) to pick an action from
- The controller is also an MLP that represents a policy, inputs: current state and memory encoding, output:
action
    - This is used for both imagination and execution
- The Model is an interaction network (https://arxiv.org/pdf/1612.00222.pdf), input: state and action, output
next state and reward aka it maps states and actions to next states and rewards
- The model, manager and controller/memory are split while training, aka they are trained at the same time
separately using different methods (ex is manager uses REINFORCE), the losses are both task loss and resource loss
aka it tries to not only optimize task performance but also computation time at the same time, but different parts
might only care about one or the other (controller and memory only train on the task loss)
- Note: The way the imagination part before executing an action or seq of actions (if the manager continues to
pick 'act') affects how the controller actually picks the actions in the real env is through the memory, the
controller when executing does not explicitly pick like a plan or something, but when the imagination/planning
part was happening all the info it got was sent into the memory LSTM and so it was encoded, then this info
is sent into the controller at each execution time step along with the current state to make it so it follows
the best path it found implicitly (aka through the encoding as it holds more info about what was just planned)
while executing. So the manager will better learn what states to imagine from and encode that, the controller while
planning will better learn how to pick actions (also applies to executing) and the memory will better learn
how to effectively encode what just was planned with the past history to allow the execution to 'go as planned'

### Imagination-Augmented Agents for Deep Reinforcement Learning (I2A)
https://arxiv.org/pdf/1707.06203.pdf
- this implements a new architecture called Imagination-Augmented Agent (I2A)
- this combines model free rl with model based rl
- it uses imagination to do it which means using an internal model to learn about the future aka not explicitly
map a plan to a policy in the real env
    - this is a reason why this works in comparison to most model based rl
    - in most model based rl the model is assumed to be perfect in the algorithm but usually it is an
    imperfect approx and therefore this leads to problems in learning, so by using it to guide the policy
    but not explicitly plan for it this method helps
- The summary is this arch gets the current state and then through makes trajectories from imagination aka using
an env model and a rollout policy it creates a set number of fixed length trajectories (makes up the imagination
core) and then encodes each traj using a rollout encoder then we aggregate all these rollout embeddings and
combine this with a model-free path to get the current policy (action prob for the current state) and state value
- Imagination Core: this is the policy rollout and the env model
    - this is given the current state and makes multiple rollouts/trajectories
    - the env model is from a new type of NN called a action-conditional next-step predictor
    - the rollout policy is a normal NN that outputs a policy (action probs)
    - the rollout encoder is an LSTM that encodes each traj backwards (aka last to first) to mimic bellman
    backups (DP)
- We then use an aggregator (NN) to combine these into a single representation
- A model free path is created (using a normal model free agent) and this is combined with the aggregation
to output the main policy and state value
- The rollout policy is trained in such a way that it is to become similar to the main policy each time, this
is through using a cross entropy kind of loss between the two policies in such a way that makes the rollout
policy similar to the main policy, and this loss is added to the total loss as an aux loss
    - this main reason for this is because we want the rollout policy to also learn from the whole policy
    in becoming better at imagining
    - this is like the other imagination based one above cuz in that one it was the same policy that was used
- The env model can be trained as we go or pre trained (pre is better), and so you can change how much the
imagination based part affects the main policy based on how much pretraining the env model got
- Using the main performance loss and these aux losses and maybe a resource loss we use A3C as the training alg
on this architecture (this works cuz the arch outputs the policy and state value aka it can be used as an
actor-critic model which is how it is trained)


## Multi-Agent RL
### Multi-Agent Deep Reinforcement Learning (MADRL)
http://cs231n.stanford.edu/reports/2016/pdfs/122_Report.pdf
- This algorithm (MADRL) shows how to train agents in a multi-agent setting
- Specifically it works for cooperation (team-based) using the pursuit-evasion domain
- One of the challenges for multi-agent RL is the training, this algorithm works by only training one agent
at a time and after a set number of steps distributing (copy) its learning policy to the agents on its team
- The reason you cannot just have the other agents count as part of the env when training a certain one is cuz
it would increase the stochasticity of the env if training all agents at the same time and in relation to this
make it very hard for an agent to figure out the dynamics of the env with other intelligent agents counted
as part of it
- This method also allows for scalability for any number of agents cuz training cost would increase only
feedforward cost for selecting an action (uses DQN)


## Misc.
### Universal Value Function Approximators
http://proceedings.mlr.press/v37/schaul15.pdf
- UVFA: V(s,g; param) aka it's just like approx a value per state but with a goal added in too, it uses a dif
reward func for the imm reward which is just based on achieving the goal, aka i guess the imm rewards are no
longer based on the env extrinsic rewards but only on the goal completion
- The goals are handcrafted per domain along with the imm reward functions
- The best way to train a nn to use this new value function is to embed the states (1 hot) and the goals (1 hot)
(means its tabular in this ex) like you would word2vec, then when going through an episode combine the
embeddings of the state you are at and the current goal through an MLP to figure out the V of said s and g
w.r.t params (since many times the state and goal space is far too big we build up a table as we go and embed
the states and goals, separately with their embedding networks, and then combine)
- The point of this paper it more so to show a new kind of value func approx alg can use, it in and of itself
is not really an alg, but it can be used in alg that sub goals would help with in the case of enhancing
exploration and finding sparse rewards

### A Simple Neural Network Module for Relational Reasoning (RN)
https://arxiv.org/pdf/1706.01427.pdf
- This shows the structure of the Relation Network (RN) in solving complex relational problems
- You have a MLP g that takes as input a pair of objects (outputs a normal vector), you send in all combos
of object pairs into g one at a time and then sum up the output for each one into a single vector, then
send this as input into a MLP f which then outputs a softmax distribution over the number of classes which
represents the answer to the question (stochastic obviously cuz softmax aka it's normal supervised learning)
    - the g MLP represents the 'relation' in each object pair
    - the f MLP then takes the sum of all these pair relations and finds the meaningful relationship
    between all these pair relationships to answer the question
- So to answer questions that have a picture we use a CNN to process the picture and from the last feature maps
created (no dense layers) we grid it up into cells and then each cell (3d cell cuz multiple feature maps in
a stack) and represent each as a vector that represents an object (aka the CNN must learn to really find
objects and process them as features)
- We also have a LSTM to process and encode the question into a vector, we send this vector into each iteration
of the g MLP along with each obj pair (same question vector is sent in each time obviously) so it learns to find
the relationships between the two objects with respect to the question at hand
- The final output of the f MLP is calculated using cross-entropy (classification) and this is back propagated
through each network in the architecture (aka f, g, CNN and LSTM in no particular order cuz they are each
separate technically, their outputs just feed into one another, it's not a parameterized connection)
- This is the new state-of-the-art alg/arch for relational reasoning/learning

### Prioritized Experience Replay
https://arxiv.org/pdf/1511.05952.pdf
- Instead of uniformly randomly picking which exp to replay from the memory use stochastic prioritization based
off of the most recent TD Error for that transition
- So the probability of picking each transition is proportional to their TD Error (saved with the transition)
- Initially set its priority to the max one it can have (when it's first made and added to the memory since it
won't have TD Error yet for the priority to be equal to) and update its current TD Error/Priority each time it
is replayed
- So you have to actually use importance sampling aka multiply an importance sampling weight to the priority
before updating the agent params when you replay a transition
- So bias is introduced to picking the exp to replay in a non uniformly random way
- This is because the expectation’s distribution (aka the estimation of the value of a state or s-a pair) is based
off samples generated from running the agent through the env aka it's based off of the dynamics of the env
and the randomness (hence expectation and not just being equal to) so if you train the agent based off of
a different distribution of picking which transitions to use than the distribution of the expectation than you
are actually biasing it to train the agent based off of a wrong representation of the env dynamics which means
it will actually converge to a different solution than the one actually represented by the problem and what
the expectation would have been has it not been for a biased exp replay updating
- So if you multiply the TD Error (aka how you're updating the agent) by the inverse of the priority thank you
can help decrease the bias a bit
- So also as the agent continues in an episode it will get more and more off in its choices if you aren't
using IS (or enough of it) cuz it will get more and more off a trajectory that
would actually occur given a non-biased picking of training transitions, so annealing is a thing to use
which just means increase the amount of IS weighing you are doing (update less and less towards the dir
and mag of the TD Error as you go since it will be getting more and more off since its training as it goes)


## Business Management
### Recurrent Reinforcement Learning: A Hybrid Approach
https://arxiv.org/pdf/1509.03044.pdf
- An LSTM was used as the NN for this algorithm to learn the states of a POMDP
- Supervised Learning (LSTM) and Reinforcement Learning (DQN) Hybrid
- The current observation is fed into the LSTM which outputs the predicted next observation and predicted reward,
and uses the DQN to output the Q-val for the state-action pair
- The predicted next observation and predicted reward signals are compared with the actual next observation
and reward and through this supervised step the differences are compared and the LSTM params are shifted
- And the DQN predicted Q-value is compared with the bellman eq per normal and trained per a normal DQN RL step
- The domain was CRM or Customer Relationship Management and the goal was LTV or Lifetime Value of customers
- A specific data set was used, and the authors made a simulator to create synthetic training data using the
original data set
- The observations were things like donation amount, frequency and stuff like that with the reward being if
they donated in the next some odd amount of time
- There was 23 steps per donor tracked (like 23 steps in a game) and batches of these 23 steps for varying
numbers of donors were fed into the LSTM + DQN model
- This mainly shows how to make a hybrid algorithm (SL + RL) but also shows how it might be applied to CRM data
in a synthetic online way, using the crafted simulator, not in real online application


## Intelligent Transportation Systems
### Coordinated Deep Reinforcement Learners for Traffic Light Control (Under Construction)
http://www.fransoliehoek.net/docs/VanDerPol16LICMAS.pdf
- The problem is optimizing the configurations of traffic lights at intersections
- The states were visual-esc representations of an intersection (used the traffic simulator SUMO)
- The actions were selecting which lanes to turn green
- They had to craft a reward function which was an equation that consisted of 4 penalties for crashes/jams,
emergency stops, delays and wait time for every car in the intersection/lanes and a slight change penalty constant
to prevent flickering (the agent changing the configuration constantly)
- The single agent DQN did OK, but the performance oscillated heavily, either because traffic is more volatile
than atari games or because of a neural network phenomenon called 'catastrophic forgetting' which is when it
forgets earlier learned structures when trying to learn new structures/scenarios
- IMPORTANT: There was a multi-agent section after the single-agent one that is hard to understand for now


## Computer Systems
### Device Placement Optimization with Reinforcement Learning
https://arxiv.org/pdf/1706.04972.pdf
- This paper optimizes the device (CPU, GPU) placement for TF computational graphs
    - It converts the graph into a list of operations and then this alg optimizes the assignment of each
    operation to a device
- Human experts do this usually but they can make mistakes, and other algs for this dont work too well with
complex and dynamic deep neural networks
- The model is a seq-to-seq LSTM (Encoder-Decoder with an attention mechanism like the machine translation paper)
and it is trained with REINFORCE (REINFORCE is used because the model acts as the policy and assigns the
operations and then the program is run and the execution time is used as the final return, so no TD alg can be
used cuz stuff only happens at the start and there is only the final return used)
- At each step the encoder takes in an operation (each operation is embedded into a vector) and then the decoder
uses an content-based attention mech and assigns each op in order (hence the use of an LSTM) to a device
- This worked very well, 20% improvement over experts and 350% improvement over prev algs
