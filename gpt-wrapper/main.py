import os 
import gymnasium as gym 
import minari 
import numpy as np 
import openai 
from dotenv import load_dotenv
load_dotenv()



def make_mpc_prompt(state, t, last_actions=None, last_states=None):
    """
    Packages the current state (and optionally a very short history) into a text
    prompt asking gpt-4-mini to output the next action for HalfCheetah.
    """
    # Example: you might say: “At time t, the HalfCheetah’s positions and velocities are [...]. 
    # Please output the action as a list of 6 floats, e.g. [0.1, -0.02, ...].” 
    # Keep it as concise as possible so you don’t blow your token budget.
    state_list = [round(x, 3) for x in state.tolist()]
    prompt = (
        f"Time step {t}. HalfCheetah-v2 state vector has dimension 17. "
        f"Current state: {state_list}.\n"
    )
    return prompt

def query_gpt_action_json(prompt, model="gpt-4-mini", temperature=0.0, max_tokens=50):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system",  "content": "You are an expert Mujoco environment controller."},
            {"role": "user",    "content": prompt + 
                "\nRespond in strict JSON: {\"action\": [f1, f2, f3, f4, f5, f6]}. No extra text."
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
    )

    text = completion.choices[0].message.content.strip()
    try:
        data = json.loads(text)
        action_arr = np.array(data["action"], dtype=np.float32)
        if action_arr.shape != (6,):
            raise ValueError(f"Expected 6 floats, got shape {action_arr.shape}")
        return action_arr

    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw model text was:", repr(text))
        return np.zeros(6, dtype=np.float32)

def mpc_with_gpt(dataset_name="mujoco/halfcheetah/expert-v0", max_steps=100, act_dim=6, save_path=None):
    """
    Runs a roll‐out in the Gym environment, but at each step it:
     1) Builds a prompt from the current state,
     2) Calls GPT-4-mini to get the next action,
     3) Steps the env,
     4) Repeats.
    """
    breakpoint()
    # 1) Make sure OPENAI_API_KEY is visible:
    assert os.getenv("OPENAI_API_KEY") is not None, "Set OPENAI_API_KEY first!"

    dataset = minari.load_dataset(dataset_name)  
    env  = dataset.recover_environment()
    obs = env.reset(seed=0)[0]

    actions = np.zeros((0, act_dim))

    for t in range(max_steps):
        prompt = make_mpc_prompt(state=obs, t=t)
        action = query_gpt_action_json(prompt)  # shape (6,)
        actions[-1] = action 
        next_obs, reward, done, truncated, info = env.step(action)
        obs = next_obs

        if done or truncated:
            break

    env.close()

    actions_np = actions.detach().cpu().numpy()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, actions_np)
    
    env.close()

    return actions_np



if __name__ == "__main__":
    actions = mpc_with_gpt()
    
    pass 

