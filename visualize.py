import os 
import gymnasium as gym 
import torch 
import numpy as np
import minari 
from decision_transformer.models.decision_transformer import DecisionTransformer
import imageio 
import random 

os.environ["MUJOCO_GL"] = "egl"

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


def load_decision_transformer(checkpoint_path, device="cuda"):
    model_kwargs = {
        "state_dim": 17,
        "act_dim" : 6,
        "max_length" : 64, 
        "max_ep_len": 1000,
        "hidden_size": 128,
        "n_layer": 3,
        "n_head": 1,
        "n_inner": 4*128,
        "activation_function": "relu", 
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1, 
    }

    model = DecisionTransformer(**model_kwargs)

    checkpoint_items = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_items['model_state_dict'])
    #model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    
    model_summary(model)

    return model 

@torch.no_grad
def rollout_dt_actions(dataset_name, model, device="cuda", save_path=None, state_dim=17,\
    act_dim=6, max_ep_len=1000, scale=1000, target_rew=600):
    model.eval()
    model.to(device=device)

    target_return = target_rew / scale 

    dataset = minari.load_dataset(dataset_name)
    env  = dataset.recover_environment()
    state = env.reset(seed=0)[0]

    state_mean = np.array([-7.60008042e-02,  5.19326348e-02,  1.29674262e-01, -4.86306923e-02,
        9.46252135e-02, -6.95297349e-02,  1.05893414e-01,  3.70562540e-02,
        1.66194899e+01,  1.25780093e-01, -1.39131576e-01,  1.11608620e+00,
       -9.88240996e-01, -5.28943071e-01,  4.83638589e-01,  1.06778732e-02,
       -2.89304906e-01])
    
    state_std = np.array([0.05319025,  0.41326522,  0.58204719,  0.56683184,  0.43607861,
        0.61718716,  0.53542789,  0.35408052,  3.54998801,  0.87528747,
        1.19326582, 11.97775664, 15.10224896,  8.62336866, 13.76872417,
       11.99385883,  7.73252534])
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

       
        pred_return = target_return[0,-1] - (reward/scale)
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    actions_np = actions.detach().cpu().numpy()
    if save_path is not None:
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, actions_np)
    
    env.close()

    return actions_np

def replay_offscreen(dataset_name, action_array, fps=30, out_path="cheetah.mp4"):
    dataset = minari.load_dataset('mujoco/halfcheetah/expert-v0')
    env  = dataset.recover_environment(render_mode="rgb_array")

    #env = gym.make(env_name, render_mode="rgb_array")
    state = env.reset(seed=0)[0]

    frames = []
    for act in action_array:
        state, reward, done, truncated, info = env.step(act)
        # env.render() returns an H×W×3 uint8 array
        frame = env.render()  
        frames.append(frame)
        if done or truncated:
            break

    env.close()

    # Write to an MP4 file with your chosen fps
    imageio.mimwrite(out_path, np.asarray(frames), fps=fps)
    print(f"Saved video to {out_path}")

if __name__ == "__main__":
    dt_model = load_decision_transformer('/home/ubuntu/small-llm/test-decision-transformer/ckpts/best_dt_0056.pth')
    np_actions = rollout_dt_actions('mujoco/halfcheetah/expert-v0', dt_model, max_ep_len=100)

    replay_offscreen('mujoco/halfcheetah/expert-v0', np_actions, out_path=os.path.join("/home/ubuntu/small-llm/test-decision-transformer/saved_vids", f"dt_cheetah_{random.randint(0,100000)}.mp4"))


