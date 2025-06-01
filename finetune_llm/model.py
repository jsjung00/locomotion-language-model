from transformers import AutoConfig, AutoModelForCausalLM
import torch.nn as nn 
import torch 

class NoRotaryEmbedding(nn.Module):
    def forward(self, q, seq_len=None):
        # Return dummy cos/sin tensors of the right shape so later code works
        dtype, device = q.dtype, q.device
        cos = torch.ones_like(q, dtype=dtype, device=device)
        sin = torch.zeros_like(q, dtype=dtype, device=device)
        return cos, sin

class DTPythia(nn.Module):
    def __init__(self,
                pretrained_model_id,
                state_dim,
                act_dim,
                max_length=None,
                max_ep_len=4096,
                action_tanh=True,
                **kwargs
                ):
        super().__init__()

        self.state_dim = state_dim 
        self.act_dim = act_dim 
        self.max_length = max_length
        self.max_ep_len = max_ep_len
        self.action_tanh = action_tanh

        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_model_id)
        cfg = self.lm.config 

        self.lm.gpt_neox.embed_in =  nn.Identity()
        self.lm.gpt_neox.emb_dropout = nn.Identity()
        self.lm.embed_out = nn.Identity() 

        # freeze all parameters in backbone
        self.lm.requires_grad_(False)
        for p in self.lm.gpt_neox.parameters():
            p.requires_grad = False

        #self.lm.gpt_neox.rotary_emb = NoRotaryEmbedding()

        hidden_size = cfg.hidden_size
        self.hidden_size = hidden_size 

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_return = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embed_state = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embed_action = nn.Sequential(
            nn.Linear(self.act_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.embed_ln = nn.LayerNorm(hidden_size)
        
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )



    @staticmethod
    def _unfreeze(*modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # use pythia backbone
        model_out = self.lm.gpt_neox(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )
      
        h = model_out.last_hidden_state

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = h.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        #return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        #state_preds = self.predict_state(x[:,2])    # predict next state given state and action #TODO: bug here? only extract the second dim 
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return None, action_preds, None 

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]




