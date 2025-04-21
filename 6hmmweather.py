import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


states = ["sunny", "cloudy", "rainy"]
trans_mat = np.array([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]])
means = {"sunny":[30,30],"cloudy":[22,50],"rainy":[18,80]}
covs = {"sunny":[[5,2],[2,3]],"cloudy":[[4,1],[1,4]],"rainy":[[3,1],[1,3]]}

def simulate_weather_sequence(n_days=300):
    obs, states_idx = [], []
    s = np.random.choice(len(states))
    for _ in range(n_days):
        obs.append(np.random.multivariate_normal(means[states[s]], covs[states[s]]))
        states_idx.append(s)
        s = np.random.choice(len(states), p=trans_mat[s])
    return np.array(states_idx), np.array(obs)

true_states, observations = simulate_weather_sequence()

discrete_obs = KBinsDiscretizer(n_bins=4, encode='ordinal').fit_transform(observations).astype(int)
obs_symbols = (discrete_obs[:, 0]*4 + discrete_obs[:, 1]).reshape(-1,1)

model_d = hmm.MultinomialHMM(n_components=3, n_iter=1000).fit(obs_symbols)
pred_d = model_d.predict(obs_symbols)

model_c = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=1000).fit(observations)
pred_c = model_c.predict(observations)

def map_states(true, pred):
    return np.vectorize(lambda x: mode(true[pred==x], keepdims=True).mode[0] if len(true[pred==x]) else x)(pred)

print("Discrete HMM Accuracy:", accuracy_score(true_states, map_states(true_states, pred_d)))
print("Continuous HMM Accuracy:", accuracy_score(true_states, map_states(true_states, pred_c)))