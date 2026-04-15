"""
algorithms.py
─────────────
Implémentation centralisée avec extraction complète des trajectoires physiques.
Gère formellement la séparation entre le temps d'entraînement (dt_train) 
et le temps d'évaluation (dt_eval).
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from jax.example_libraries.stax import Dense, serial, Softplus, FanInSum, FanOut, Identity, parallel

from physics import PipelineConfig, make_jax_potential

# 1. OUTILS PARTAGÉS
def build_score_network(space_dim: int, n_steps: int, emb_dim: int):
    input_dim = space_dim + emb_dim
    resnet_block = serial(FanOut(2), parallel(Identity, serial(Dense(input_dim), Softplus)), FanInSum)
    init_nn, apply_nn = serial(resnet_block, resnet_block, Dense(space_dim))

    def init_fn(rng): # Initialise les poids
        _, nn_params = init_nn(rng, (input_dim,))
        emb_table = jax.random.normal(rng, (n_steps, emb_dim)) * 0.05 # Chaque étape a son propre vecteur aléatoire.
        return {"nn": nn_params, "emb": emb_table, "scale": jnp.array(0.0)}

    def apply_fn(params, q, step_idx):
        emb = params["emb"][step_idx] # Cherche le vecteur correspondant à l'étape actuelle.
        out = apply_nn(params["nn"], jnp.concatenate([q, emb])) # Colle la position et le temps ensemble, et les fait passer dans le réseau.
        return out * params["scale"] # Comme scale vaut 0 au début, le réseau génère 0 force au début.
    
    return init_fn, apply_fn

def _sample_q0(cfg: PipelineConfig, key):
    k1, k2 = jax.random.split(key) # JAX exige de scinder la clé aléatoire pour chaque nouvelle opération afin de garantir la reproductibilité.
    comp = jax.random.choice(k1, jnp.arange(cfg.gmm0.means.shape[0]), p=jnp.array(cfg.gmm0.weights)) # Tire au sort de quelle gaussienne du mélange la particule va provenir, en respectant les poids (p=weights).
    return jnp.array(cfg.gmm0.means)[comp] + jax.random.multivariate_normal(k2, jnp.zeros(cfg.dim), jnp.array(cfg.gmm0.covs)[comp]) # Décale la particule vers le centre de la gaussienne choisie et lui applique le bruit (matrice de covariance) correspondant.

def _train_loop(loss_fn, params, cfg: PipelineConfig, desc="Entraînement"): # Optimise les poids pour minimiser loss_fn en utilisant une descente de gradient avec un calendrier de vitesse avancé.
    optimizer = optax.chain( #Combine plusieurs stratégies d'optimisation
        optax.clip_by_global_norm(cfg.clip_norm), # Empêche le réseau de faire des pas d'apprentissage trop violents si le gradient explose
        optax.adamw( # L'optimiseur Adam avec le Weight Decay (pénalité L2 sur les poids)
            learning_rate=optax.warmup_cosine_decay_schedule( # La vitesse d'apprentissage commence à 0, monte jusqu'à lr_init pendant 1000 epochs, puis redescend en courbe cosinus.
                init_value=0.001,
                peak_value=cfg.lr_init,
                warmup_steps=int(cfg.n_epochs * 0.2),
                decay_steps=cfg.n_epochs,
                end_value=0.00005
            ),
            weight_decay=cfg.weight_decay
        )
    )
    opt_state = optimizer.init(params) # Initialise la mémoire de l'optimiseur
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=1)) # Calcule automatiquement la dérivée de la fonction de perte par rapport aux poids
    eval_loss_fn = jax.jit(loss_fn)
    key = jax.random.PRNGKey(cfg.seed)
    loss_history = []
    best_loss, patience_ctr = float("inf"), 0

    pbar = tqdm(range(cfg.n_epochs), desc=desc)
    for epoch in pbar:
        key, sk = jax.random.split(key)
        seeds = jax.random.randint(sk, (cfg.batch_size_train,), 1, 1000000) # Crée un lot de graines aléatoires (pour générer le lot de trajectoires).
        grads = grad_fn(seeds, params) # Calcule la direction de l'erreur
        updates, opt_state = optimizer.update(grads, opt_state, params) # Calcule comment modifier les poids.
        params = optax.apply_updates(params, updates) # Applique la modification aux params.
        
        if epoch % 50 == 0: # Gère l'affichage, sauvegarde best_loss, et déclenche early stop si le réseau n'apprend plus rien après le temps imparti par la patience.
            
            #current_loss = float(loss_fn(seeds, params))
            current_loss = float(eval_loss_fn(seeds, params))
            pbar.set_postfix(loss=f"{current_loss:.3f}")
            loss_history.append(current_loss)


            best_loss, patience_ctr = float("inf"), 0
            best_params = params # Initialisation
            if current_loss < best_loss:
                best_loss, patience_ctr = current_loss, 0
                best_params = params # On sauvegarde l'état optimal
            else:
                patience_ctr += 1
            """
            if current_loss < best_loss:
                best_loss, patience_ctr = current_loss, 0
            else:
                patience_ctr += 1
            """ 

            if patience_ctr >= cfg.patience // 50:
                pbar.set_description(f"{desc} [early stop @ {epoch+1}]")
                break
                
    #return params, loss_history
    return best_params, loss_history

def _batched_eval(compute_eval_fn, cfg: PipelineConfig): # Génère 10 000 trajectoires finales pour les graphiques, en découpant le travail en petits lots (RAM)
    """Évalue W et compile les trajectoires complètes."""
    compute_batch = jax.jit(jax.vmap(compute_eval_fn))
    works, trajs = [], []
    key = jax.random.PRNGKey(cfg.seed + 99)
    
    for _ in tqdm(range(0, cfg.n_samples, cfg.batch_size_val), desc="Évaluation (Extraction Trj.)", leave=False):
        key, sk = jax.random.split(key)
        seeds = jax.random.randint(sk, (cfg.batch_size_val,), 1, 1000000)
        w_b, traj_b = compute_batch(seeds) # Calcule le Travail et les trajectoires du lot
        works.append(w_b)
        trajs.append(traj_b)
        
    return np.concatenate(works)[:cfg.n_samples], np.concatenate(trajs)[:cfg.n_samples] #Colle tous les blocs ensemble pour renvoyer le tableau final des 10 000 particules

# 2. ALGORITHMES ==========================================================================================================================
def run_ais(cfg: PipelineConfig):
    V, grad_V, dV_dlam, _, _ = make_jax_potential(cfg.gmm0, cfg.gmm1)
    
    # AIS utilise strictement la haute résolution (évaluation)
    dt_eval = cfg.dt_eval
    sched_eval = jnp.array(cfg.schedule_eval)

    def compute_eval(seed): # Définit ce qui arrive à une particule.
        q0 = _sample_q0(cfg, jax.random.PRNGKey(seed))
        
        def step(carry, k): # La fonction exécutée à chaque étape k.
            q, w, key = carry
            w += (sched_eval[k+1] - sched_eval[k]) * dV_dlam(q, sched_eval[k])
            key, sk = jax.random.split(key)
            q_new = q - dt_eval * grad_V(q, sched_eval[k+1]) + jnp.sqrt(2.0 * dt_eval) * jax.random.normal(sk, (cfg.dim,))
            return (q_new, w, key), q_new

        (_, w_final, _), q_hist = jax.lax.scan(step, (q0, 0.0, jax.random.PRNGKey(seed)), jnp.arange(cfg.n_steps_eval))
        full_traj = jnp.concatenate([q0[None, ...], q_hist], axis=0) # Ajouter la position init q0
        return w_final, full_traj

    return _batched_eval(compute_eval, cfg)

# --------------------------------------------------------------------------------------------------

def train_and_eval_mcd(cfg: PipelineConfig):
    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    # L'IA est dimensionnée sur n_steps_train
    init_fn, apply_fn = build_score_network(cfg.dim, cfg.n_steps_train, cfg.emb_dim)
    params = init_fn(jax.random.PRNGKey(cfg.seed))

    def make_elbo_fn(dt, sched, n_steps, is_eval=False):
        sched = jnp.array(sched)
        def elbo_single(seed, params):
            q0 = _sample_q0(cfg, jax.random.PRNGKey(seed))      

            def step(carry, k):
                q, acc, key = carry # acc = elbo accumulé
                key, sk = jax.random.split(key)
                gq = grad_V(q, sched[k]) # utilisée 2 fois calculée 1 fois
                q_new = q - dt * gq + jnp.sqrt(2.0 * dt) * jax.random.normal(sk, (cfg.dim,))
                
                log_f = -jnp.sum((q_new - q + dt * gq)**2) / (4.0 * dt) # c'est redondant mais c'est plus lisible

                # Résolution de l'index temporel
                if is_eval:
                    k_train = jnp.minimum(jnp.floor(k * dt / cfg.dt_train).astype(int), cfg.n_steps_train - 1)
                else:
                    k_train = k

                score = apply_fn(params, q_new, k_train)

                mean_b = q_new + dt * grad_V(q_new, sched[k]) + 2.0 * dt * score
                log_b = -jnp.sum((q - mean_b)**2) / (4.0 * dt) # proba que q = q_prédite
                
                return (q_new, acc + log_b - log_f, key), q_new

            (_, acc_K, _), q_hist = jax.lax.scan(step, (q0, 0.0, jax.random.PRNGKey(seed)), jnp.arange(n_steps))
            full_traj = jnp.concatenate([q0[None, ...], q_hist], axis=0)
            return log_g1(q_hist[-1]) - log_g0(q0) + acc_K, full_traj # on ajoute les dernieres proba (début et fin)
        return elbo_single

    # Phase d'Entraînement
    elbo_train = make_elbo_fn(cfg.dt_train, cfg.schedule_train, cfg.n_steps_train, is_eval=False)
    def loss_fn(seeds, params):
        return -jnp.mean(jax.vmap(lambda s: elbo_train(s, params)[0])(seeds))
    params, losses = _train_loop(loss_fn, params, cfg, desc="Train MCD")
    
    # Phase d'Évaluation
    elbo_eval = make_elbo_fn(cfg.dt_eval, cfg.schedule_eval, cfg.n_steps_eval, is_eval=True)
    def compute_eval(seed):
        elbo, traj = elbo_eval(seed, params)
        return -elbo, traj # W = -ELBO
        
    works, trajs = _batched_eval(compute_eval, cfg)
    return (works, trajs), losses

# --------------------------------------------------------------------------------------------------

def train_and_eval_cmcd(cfg: PipelineConfig):
    V, grad_V, _, log_g0, log_g1 = make_jax_potential(cfg.gmm0, cfg.gmm1)
    Lp = 1.0 / cfg.T
    init_fn, apply_fn = build_score_network(cfg.dim, cfg.n_steps_train, cfg.emb_dim)
    params = init_fn(jax.random.PRNGKey(cfg.seed))

    def make_elbo_fn(dt, sched, n_steps, is_eval=False):
        sched = jnp.array(sched)
        def elbo_single(seed, params):
            q0 = _sample_q0(cfg, jax.random.PRNGKey(seed))
            
            def step(carry, k):
                q, acc, key = carry
                # Index temporel pour l'Aller
                if is_eval:
                    k_aller = jnp.minimum(jnp.floor(k * dt / cfg.dt_train).astype(int), cfg.n_steps_train - 1)
                else:
                    k_aller = k  
                u_q = apply_fn(params, q, k_aller)
                key, sk = jax.random.split(key)
                gq = grad_V(q, sched[k])
                q_new = q - dt * gq + dt * Lp * u_q + jnp.sqrt(2.0 * dt) * jax.random.normal(sk, (cfg.dim,))
                
                log_f = -jnp.sum((q_new - q + dt * gq - dt * Lp * u_q)**2) / (4.0 * dt)

                # Index temporel pour le Retour (k+1)
                if is_eval:
                    k_retour = jnp.minimum(jnp.floor((k + 1) * dt / cfg.dt_train).astype(int), cfg.n_steps_train - 1)
                else:
                    # k_retour = k + 1
                    k_retour = jnp.minimum(k + 1, cfg.n_steps_train - 1)
                u_qnew = apply_fn(params, q_new, k_retour)
                mean_b = q_new - dt * grad_V(q_new, sched[k+1]) - dt * Lp * u_qnew # Signe (-) validé
                log_b = -jnp.sum((q - mean_b)**2) / (4.0 * dt)
                
                return (q_new, acc + log_b - log_f, key), q_new

            (_, acc_K, _), q_hist = jax.lax.scan(step, (q0, 0.0, jax.random.PRNGKey(seed)), jnp.arange(n_steps))
            full_traj = jnp.concatenate([q0[None, ...], q_hist], axis=0)
            return log_g1(q_hist[-1]) - log_g0(q0) + acc_K, full_traj
        return elbo_single

    # Phase d'Entraînement
    elbo_train = make_elbo_fn(cfg.dt_train, cfg.schedule_train, cfg.n_steps_train, is_eval=False)
    def loss_fn(seeds, params):
        return -jnp.mean(jax.vmap(lambda s: elbo_train(s, params)[0])(seeds))
    params, losses = _train_loop(loss_fn, params, cfg, desc="Train CMCD")
    
    # Phase d'Évaluation
    elbo_eval = make_elbo_fn(cfg.dt_eval, cfg.schedule_eval, cfg.n_steps_eval, is_eval=True)
    def compute_eval(seed):
        elbo, traj = elbo_eval(seed, params)
        return -elbo, traj
        
    works, trajs = _batched_eval(compute_eval, cfg)
    return (works, trajs), losses

# --------------------------------------------------------------------------------------------------

def train_and_eval_led(cfg: PipelineConfig):
    V, grad_V, dV_dlam, _, _ = make_jax_potential(cfg.gmm0, cfg.gmm1)
    Lp = 1.0 / cfg.T
    init_fn, apply_fn = build_score_network(cfg.dim, cfg.n_steps_train, cfg.emb_dim)
    params = init_fn(jax.random.PRNGKey(cfg.seed))
    
    def make_work_fn(dt, sched, n_steps, is_eval=False):
        sched = jnp.array(sched)
        def work_single(seed, params):
            q0 = _sample_q0(cfg, jax.random.PRNGKey(seed))
            
            def step(carry, k):
                q, w, key = carry
                
                # Résolution de l'index temporel
                if is_eval:
                    k_train = jnp.minimum(jnp.floor(k * dt / cfg.dt_train).astype(int), cfg.n_steps_train - 1)
                else:
                    k_train = k
                    
                u_fn = lambda qq: apply_fn(params, qq, k_train)
                u_q = u_fn(q)

                div_u = jnp.trace(jax.jacfwd(u_fn)(q))
                
                lam = sched[k]
                dlam = sched[k+1] - sched[k]
                g_q = grad_V(q, lam)
                
                w += dlam * (dV_dlam(q, lam) + jnp.sum(u_q * g_q) - div_u)
                
                key, sk = jax.random.split(key)
                q_new = q - dt * g_q + dt * Lp * u_q + jnp.sqrt(2.0 * dt) * jax.random.normal(sk, (cfg.dim,))
                
                return (q_new, w, key), q_new

            (_, w_K, _), q_hist = jax.lax.scan(step, (q0, 0.0, jax.random.PRNGKey(seed)), jnp.arange(n_steps))
            full_traj = jnp.concatenate([q0[None, ...], q_hist], axis=0)
            return w_K, full_traj
        return work_single

    # Phase d'Entraînement
    work_train = make_work_fn(cfg.dt_train, cfg.schedule_train, cfg.n_steps_train, is_eval=False)
    def loss_fn(seeds, params):
        works = jax.vmap(lambda s: work_train(s, params)[0])(seeds)
        return jnp.mean(works)
    params, losses = _train_loop(loss_fn, params, cfg, desc="Train LED")
    
    # Phase d'Évaluation
    work_eval = make_work_fn(cfg.dt_eval, cfg.schedule_eval, cfg.n_steps_eval, is_eval=True)
    def compute_eval(seed):
        w_final, traj = work_eval(seed, params)
        return w_final, traj
        
    works, trajs = _batched_eval(compute_eval, cfg)
    return (works, trajs), losses