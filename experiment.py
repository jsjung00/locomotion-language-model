import gymnasium as gym
import minari 
import numpy as np
import torch
import wandb

from tqdm import tqdm 

import argparse
import pickle
import random
import sys
import time
import os 
from visualize import load_decision_transformer, load_pythia

from finetune_llm.model import DTPythia

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

def model_summary(model):
    print("="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            status = "✓"
        else:
            status = "✗"
            
        print(f"{status} {name:30} {str(param.shape):20} {param_count:>10,}")
    
    print("="*50)
    print(f"Total parameters:     {total_params:>10,}")
    print(f"Trainable parameters: {trainable_params:>10,}")
    print(f"Non-trainable params: {(total_params - trainable_params):>10,}")
    print("="*50)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'halfcheetah':
        dataset = minari.load_dataset('mujoco/halfcheetah/expert-v0')
        env  = dataset.recover_environment()
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    else:
        raise NotImplementedError

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # save all path information into separate lists
    mode = 'normal'
    states, traj_lens, returns = [], [], []
    
    for episode in dataset.iterate_episodes():
        states.append(episode.observations[:-1])
        traj_lens.append(len(episode.observations))
        returns.append(episode.rewards.sum())

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)
    
    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(dataset) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
        )

        states, actions, rewards, dones, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = dataset[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj.rewards.shape[0] - max_len)

            # get sequences from dataset
            states.append(traj.observations[:-1][si:si + max_len].reshape(1, -1, state_dim))
            actions.append(traj.actions[si:si + max_len].reshape(1, -1, act_dim))
            rewards.append(traj.rewards[si:si + max_len].reshape(1, -1, 1))
            dones.append(traj.terminations[si:si + max_len].reshape(1, -1))
         
            timesteps.append(np.arange(si, si + states[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj.rewards[si:], gamma=1.)[:states[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= states[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = states[-1].shape[1]

            states[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), states[-1]], axis=1)
            states[-1] = (states[-1] - state_mean) / state_std
            actions[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., actions[-1]], axis=1)


            rewards[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rewards[-1]], axis=1)
            dones[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, dones[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        states = torch.from_numpy(np.concatenate(states, axis=0)).to(dtype=torch.float32, device=device)
        actions = torch.from_numpy(np.concatenate(actions, axis=0)).to(dtype=torch.float32, device=device)
        rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).to(dtype=torch.float32, device=device)
        dones = torch.from_numpy(np.concatenate(dones, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return states, actions, rewards, dones, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in tqdm(range(num_eval_episodes)):
                with torch.no_grad():
                    if model_type == 'dt' or model_type == "pythia":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == "pythia":
        model = DTPythia(
            pretrained_model_id='EleutherAI/pythia-410m',
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len
        )

    elif model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    model_summary(model)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == "pythia" or model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            eval_every_n_epochs=10
        )

    eval_only = variant.get('eval', False)

    if not eval_only:
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='decision-transformer',
                config=variant
            )
            # wandb.watch(model)  # wandb has some bug


        ts = time.strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join("ckpts", ts)
        os.makedirs(run_folder, exist_ok=True)

        best_loss = float('inf')
        best_path = None 

        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
            action_error = outputs['training/action_error']
            if action_error < best_loss:
                best_loss = action_error
                filename  = f"best_{model_type}_{iter+1:04d}.pth"
                best_path = os.path.join(run_folder, filename)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iter': iter + 1,
                }, best_path)

            if log_to_wandb:
                wandb.log(outputs)
    else:
        # evaluate the model 
        reward_target = variant.get('reward_target', 1200)
        eval_fn = eval_episodes(reward_target)
        

        if model_type == "dt":
            model = load_decision_transformer('/home/ubuntu/small-llm/test-decision-transformer/ckpts/best_dt_0056.pth')  
        elif model_type == "pythia":
            model = load_pythia("/home/ubuntu/small-llm/test-decision-transformer/ckpts/20250601_032836/best_pythia_0033.pth") 
        else:
            raise ValueError("only accept dt and pythia")
            pass

        outputs = eval_fn(model)
        for k, v in outputs.items():
            print(f'evaluation/{k} = {v}')

        return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='pythia')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--num_steps_per_iter', type=int, default=100) # 100
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--eval', default=False)
    parser.add_argument('--reward_target', type=int, default=1200)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
