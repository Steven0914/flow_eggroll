import jax
import optax
import jax.numpy as jnp
from .base_noiser import Noiser

from functools import partial

import optax

def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    lora_params = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), (a+b, frozen_noiser_params["rank"]), dtype=param.dtype)
    B = lora_params[:b] # b x r
    A = lora_params[b:] # a x r

    # update is A @ B.T
    return A * sigma, B

def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    updates = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), param.shape, dtype=param.dtype)
    return updates * sigma

def _simple_full_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    if frozen_noiser_params["freeze_nonlora"]:
        return jnp.zeros_like(param)
    _, thread_ids = iterinfo
    sigma = jnp.where(thread_ids % 2 == 0, base_sigma, -base_sigma)
    updates = jax.vmap(partial(get_nonlora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma, iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    broadcasted_sigma = jnp.reshape(sigma, sigma.shape + (1,) * len(param.shape))
    # return jnp.astype(jnp.mean(broadcasted_scores * updates / broadcasted_sigma ** 2, axis=0), param.dtype)
    return jnp.astype(jnp.mean(broadcasted_scores * updates, axis=0), param.dtype)

def _simple_lora_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    A, B = jax.vmap(partial(get_lora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,1))
    A = broadcasted_scores * A # N x a x r for A vs N x b x r for B -> final update is just a x b
    num_envs = scores.shape[0]
    print("LORA UPDATE", A.shape, B.shape)
    # return A.T @ B / num_envs
    return jnp.einsum('nir,njr->ij', A, B) / num_envs

def _noop_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    return jnp.zeros_like(param)

class EggRoll(Noiser):
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, **kwargs):
        """
        Return frozen_noiser_params and noiser_params
        """
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)
        
        return {"group_size": group_size, "freeze_nonlora": freeze_nonlora, "noise_reuse": noise_reuse, "solver": true_solver, "rank": rank}, {"sigma": sigma, "opt_state": opt_state}
    
    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ A @ B.T

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        # return param[x]
        raise NotImplementedError("Embedding is not implemented")

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        group_size = frozen_noiser_params["group_size"]
        if group_size == 0:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
        else:
            group_scores = raw_scores.reshape((-1, group_size))
            true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
            true_scores = true_scores.ravel()
        # fitness = jax.nn.softmax(true_scores)
        # return fitness * raw_scores.size
        return true_scores

    @classmethod
    def _do_update(cls, param, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params, **kwargs):
        update_fn = [_simple_full_update, _simple_lora_update, _noop_update, _noop_update][map_classification]

        if len(base_key.shape) == 0:
            new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            new_grad = jax.lax.scan(lambda _, x: (0, update_fn(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)), 0, xs=(param, base_key))[1]

        # return (param + new_grad * lr * jnp.sqrt(fitnesses.size)).astype(param.dtype)
        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        new_grad = jax.tree.map(lambda p, k, m: cls._do_update(p, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params), params, base_keys, es_map)
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(new_grad, noiser_params["opt_state"], params)
        return noiser_params, optax.apply_updates(params, updates)
