import torch
from modules.networks.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)

class DTQNUtils:
    def __init__(self, obs_dim, num_actions, action_dim, inner_embed_size):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.inner_embed_size = inner_embed_size
        self.discrete = False

        obs_output_dim = inner_embed_size - action_dim
        if action_dim > 0:
            # assert 1==0,f"num_actions:{action_dim}"
            self.action_embedding = ActionEmbeddingRepresentation(
                num_actions=self.num_actions, action_dim=self.action_dim
            )
        else:
            self.action_embedding = None

        # Initialize observation embedding based on obs_dim
        self.obs_embedding = self.initialize_obs_embedding(obs_output_dim)

    def initialize_obs_embedding(self, obs_output_dim):
        if isinstance(self.obs_dim, tuple):
            return ObservationEmbeddingRepresentation.make_image_representation(
                obs_dim=self.obs_dim, outer_embed_size=obs_output_dim
            )
        elif self.discrete:
            return ObservationEmbeddingRepresentation.make_discrete_representation(
                vocab_sizes=self.vocab_sizes,
                obs_dim=self.obs_dim,
                embed_per_obs_dim=self.embed_per_obs_dim,
                outer_embed_size=obs_output_dim,
            )
        else:
            return ObservationEmbeddingRepresentation.make_continuous_representation(
                obs_dim=self.obs_dim, outer_embed_size=obs_output_dim
            )

    def get_obs_embeddings(self, obss):
        """
        获取观察的嵌入。

        Args:
            obss: 输入的观察，形状为 [batch_size, context_len, n_agents, obs_dim]

        Returns:
            obs_embeddings: 观察的嵌入，形状为[batch_size, context_len, n_agents, embedding_size]
        """
        return self.obs_embedding(obss)

    def get_action_embeddings(self, actions):
        """
        获取动作的嵌入。

        Args:
            actions: 输入的动作，形状为 [batch_size, context_len,n_agents, 1]

        Returns:
            action_embeddings: 动作的嵌入，形状为 [batch_size, context_len, action_embed_dim]
        """
        if self.action_embedding is not None:
            action_embeddings = self.action_embedding(actions)

            return action_embeddings
        else:
            return None  