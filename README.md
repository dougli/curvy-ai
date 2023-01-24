# curvy-ai

A silly personal project I worked on to try to understand (and implement) PPO in a multi-agent reinforcement learning setting.

One of my coworkers was very good at a game called [Curve Fever](https://curvefever.pro/) that could be played through the browser. It's an adversarial
tron-like game, where players pilot a little craft with an increasingly long tail that serves as a wall. Players must aim to avoid hitting any walls.

We'd play this game every Friday and he frequently beat all of us. I'm a software engineer by trade and I have no ML experience, but I thought, what 
the heck, I'll see if I can train something to beat him. At the very least, I wanted to learn and see if it was possible to "cheat" on an online game
by training an agent.

## Details

**The code is awful and buggy. This was a personal project and I optimised for speed of learning and results. While I tried to keep things reasonably
structured, there is a lot of copy-pasta and you peruse it at your own risk.**

The rough project contains the following parts to make it all work together.

1. Implementation of PPO pulled from stable_baselines3, modified to run async so I can match timings with browser screenshots. I first rolled my own
version of PPO, but then went with stable_baselines3 because when I benchmarked my own version, it didn't match up to the scores in the PPO
paper. I used the neural network architecture listed in the [IMPALA paper](https://arxiv.org/pdf/1802.01561.pdf) as there's 
[evidence](https://openai.com/blog/quantifying-generalization-in-reinforcement-learning/) that it outperforms the one in the 
[Nature paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). This was true when I had briefly trained
both models up to 1.5 million timesteps.
2. Some benchmarking code against Atari breakout. I added this to make sure I didn't introduce subtle bugs while making the async PPO algorithm work.
3. Browser automation using [Pyppeteer](https://github.com/pyppeteer/pyppeteer) to start the game. This gave me access to the low level
[Chrome DevTools API](https://chromedevtools.github.io/devtools-protocol/) which I otherwise wouldn't have had.
4. OCR'd the scoring region using Tesseract to detect game start / end states and who won the game
5. Wired it all together to collect experience trajectories, run training asynchronously in the GPU. `multiprocessing` to avoid Python's global
interpreter lock.
6. Logging and output to tensorboard
7. saving previous agents to avoid strategy collapse.
8. Reward shaping to encourage the agent to live longer, as the reward signal would have too much noise otherwise.

## Brief results

After 5 million timesteps I was able to get the agent to survive for over 10 seconds and improving. For comparison, a random agent stays alive for 4.35s
on average.

![Screenshot 2023-01-24 at 21 22 26](https://user-images.githubusercontent.com/2380110/214419760-98c08598-ad4b-4b68-995c-c5847dbdc7e2.png)

I did not run this agent against human players or my coworker. There's a few reasons:

1. I had accomplished my primary goals of learning a lot, and was happy that I have a solid understanding of PPO.
2. I invested into (or obsessed over) this for many weeknights and weekends outside of my full-time job, and raising a 2 year old daughter, 
to get this all to work. At some point, I just got tired. To make this play against human players would require improvements in the browser
automation code so that it could join lobbies with humans being the host.
3. It doesn't perform at the level of an average human player, which I estimate to be 15 seconds or more. Given the training graphs, it seems 
clear that throwing further compute at the problem will lead to a better agent, but I didn't want to pay for compute.
4. It likely doesn't generalize well. While the agent plays against older versions of itself, it's settled into a strategy of
simply trying to outlive the other agent rather than aggressively trying to tackle and kill the other agent first. I wanted to investigate
batch normalization but I didn't want to spend another couple of weeks trying to make this work.
